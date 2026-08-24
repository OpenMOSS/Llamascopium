"""Real BT4 CUDA integration test for the optimized Q/K attribution path.

This test is opt-in because it loads the full model and all 15 SAE/LoRSA layers.
See the command printed in the skip message or run with ``RUN_BT4_GPU_TESTS=1``.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "server"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lm_saes.circuit.attribution_qk import merge_qk_graph
from server.circuits_service import load_model_and_transcoders, run_attribution
from server.constants import BT4_MODEL_NAME, get_bt4_sae_combo


DEFAULT_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _required_path(env_name: str, fallback: str) -> str:
    path = os.environ.get(env_name, fallback)
    if not Path(path, "L0").exists():
        pytest.fail(
            f"{env_name}={path!r} does not contain L0; point it at the 15-layer "
            "BT4 checkpoint directory."
        )
    return path


def _validate_checkpoint_kinds(tc_path: str, lorsa_path: str) -> None:
    tc_config = Path(tc_path, "L0", "config.json")
    lorsa_config = Path(lorsa_path, "L0", "config.json")
    if tc_config.exists() and "lorsa" in tc_config.read_text(encoding="utf-8").lower():
        pytest.fail(
            f"BT4_TC_BASE_PATH={tc_path!r} appears to contain LoRSA checkpoints. "
            "BT4_TC_BASE_PATH and BT4_LORSA_BASE_PATH may be swapped."
        )
    if lorsa_config.exists() and "lorsa" not in lorsa_config.read_text(encoding="utf-8").lower():
        pytest.fail(
            f"BT4_LORSA_BASE_PATH={lorsa_path!r} does not appear to contain LoRSA checkpoints. "
            "BT4_TC_BASE_PATH and BT4_LORSA_BASE_PATH may be swapped."
        )


@pytest.fixture(scope="module")
def bt4_cuda_model():
    if os.environ.get("RUN_BT4_GPU_TESTS") != "1":
        pytest.skip("set RUN_BT4_GPU_TESTS=1 to run the real BT4 CUDA test")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    combo_id = os.environ.get("BT4_SAE_COMBO", "k_30_e_16")
    combo = get_bt4_sae_combo(combo_id)
    tc_path = _required_path("BT4_TC_BASE_PATH", combo["tc_base_path"])
    lorsa_path = _required_path("BT4_LORSA_BASE_PATH", combo["lorsa_base_path"])
    _validate_checkpoint_kinds(tc_path, lorsa_path)
    model_name = os.environ.get("BT4_MODEL_NAME", BT4_MODEL_NAME)

    model, _, _ = load_model_and_transcoders(
        model_name=model_name,
        device="cuda",
        tc_base_path=tc_path,
        lorsa_base_path=lorsa_path,
        n_layers=15,
        cache_key=f"gpu-test::{model_name}::{combo_id}",
    )
    model.eval()
    return model


def test_real_bt4_qk_attribution_on_cuda(bt4_cuda_model):
    """Run one real Q+K trace and report latency plus peak allocated memory."""

    model = bt4_cuda_model
    fen = os.environ.get("BT4_GPU_TEST_FEN", DEFAULT_FEN)
    move_uci = os.environ.get("BT4_GPU_TEST_MOVE", "e2e4")
    max_features = int(os.environ.get("BT4_GPU_MAX_FEATURE_NODES", "128"))
    vjp_batch_size = int(os.environ.get("BT4_GPU_VJP_BATCH_SIZE", "8"))

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    with patch.object(model, "run_with_hooks", wraps=model.run_with_hooks) as forward_spy:
        result = run_attribution(
            model=model,
            prompt=fen,
            fen=fen,
            move_uci=move_uci,
            side="both",
            max_n_logits=1,
            desired_logit_prob=0.95,
            max_feature_nodes=max_features,
            batch_size=vjp_batch_size,
            vjp_batch_size=vjp_batch_size,
            mixed_precision_edges=True,
            order_mode="abs",
            mongo_client=None,
            sae_series="BT4-exp128",
            save_activation_info=False,
        )
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    peak_gib = torch.cuda.max_memory_allocated() / 1024**3
    expected_dtype = (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )

    # setup caches and live VJP references must come from one model forward.
    assert forward_spy.call_count == 1
    assert result["q"] is not None and result["k"] is not None
    for side in ("q", "k"):
        package = result[side]
        assert package["edge_matrix"].is_cuda
        assert package["edge_matrix"].dtype == expected_dtype
        assert package["row_to_node_index"].is_cuda
        assert package["selected_features"].is_cuda
        assert 0 < package["selected_features"].numel() <= max_features
        assert torch.isfinite(package["edge_matrix"]).all()
        # Both-side packaging intentionally avoids allocating intermediate squares.
        assert package["full_edge_matrix"] is None

    merged = merge_qk_graph(result)
    adjacency = merged["adjacency_matrix"]
    assert adjacency.is_cuda
    assert adjacency.dtype == expected_dtype
    assert adjacency.ndim == 2 and adjacency.shape[0] == adjacency.shape[1]
    assert torch.isfinite(adjacency).all()

    max_seconds = os.environ.get("BT4_GPU_MAX_SECONDS")
    if max_seconds is not None:
        assert elapsed <= float(max_seconds), (
            f"attribution took {elapsed:.2f}s, limit is {max_seconds}s"
        )
    max_peak_gib = os.environ.get("BT4_GPU_MAX_PEAK_GIB")
    if max_peak_gib is not None:
        assert peak_gib <= float(max_peak_gib), (
            f"peak allocation was {peak_gib:.2f}GiB, limit is {max_peak_gib}GiB"
        )

    print(
        "\nBT4 Q/K CUDA benchmark: "
        f"features={max_features}, vjp_batch={vjp_batch_size}, "
        f"elapsed={elapsed:.2f}s, peak_allocated={peak_gib:.2f}GiB, "
        f"edge_dtype={expected_dtype}, nodes={adjacency.shape[0]}"
    )
