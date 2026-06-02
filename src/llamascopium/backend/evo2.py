from __future__ import annotations

from contextlib import contextmanager, nullcontext
from pathlib import Path
import sys
from unittest.mock import patch

import torch
from evo2.utils import HF_MODEL_NAME_MAP

DEFAULT_EVO2_CHECKPOINTS = {
    "evo2_7b": Path("/inspire/hdd/global_user/hezhengfu-240208120186/models/evo2_7b/evo2_7b.pt"),
}
HF_TO_INTERNAL_MODEL_NAME = {hf_name: internal_name for internal_name, hf_name in HF_MODEL_NAME_MAP.items()}

_ROTARY_PATCHED = False


def _get_vendor_evo2_class():
    try:
        from evo2 import Evo2 as vendor_evo2
    except ModuleNotFoundError:
        # Allow local development before `uv sync` installs the vendored package.
        vendored_root = Path(__file__).resolve().parents[3] / "third_party" / "evo2"
        if vendored_root.exists():
            sys.path.insert(0, str(vendored_root))
        from evo2 import Evo2 as vendor_evo2
    return vendor_evo2


def _normalize_vendor_evo2_model_name(model_name: str) -> str:
    return HF_TO_INTERNAL_MODEL_NAME.get(model_name, model_name)


def _patch_rotary_for_cpu() -> None:
    global _ROTARY_PATCHED
    if _ROTARY_PATCHED or torch.cuda.is_available():
        return

    import vortex.model.rotary as model_rotary
    import vortex.ops.embedding.rotary as ops_rotary

    original_apply_rotary = ops_rotary.apply_rotary

    def cpu_safe_apply_rotary(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        seqlen_offsets=0,
        cu_seqlens=None,
        max_seqlen=None,
        interleaved=False,
        inplace=False,
        conjugate=False,
    ) -> torch.Tensor:
        if x.device.type == "cuda":
            return original_apply_rotary(
                x,
                cos,
                sin,
                seqlen_offsets=seqlen_offsets,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                interleaved=interleaved,
                inplace=inplace,
                conjugate=conjugate,
            )

        if cu_seqlens is not None:
            raise NotImplementedError("CPU Evo2 fallback does not support variable-length rotary inputs.")

        batch, seqlen, _nheads, _headdim = x.shape
        if isinstance(seqlen_offsets, torch.Tensor):
            offsets = [int(offset) for offset in seqlen_offsets.tolist()]
        else:
            offsets = [int(seqlen_offsets)] * batch

        out = x if inplace else x.clone()
        for batch_idx, offset in enumerate(offsets):
            cos_slice = cos[offset : offset + seqlen]
            sin_slice = sin[offset : offset + seqlen]
            if conjugate:
                sin_slice = -sin_slice
            rotated = model_rotary.apply_rotary_emb_torch(
                out[batch_idx : batch_idx + 1],
                cos_slice,
                sin_slice,
                interleaved=interleaved,
            )
            out[batch_idx : batch_idx + 1].copy_(rotated)
        return out

    ops_rotary.apply_rotary = cpu_safe_apply_rotary
    model_rotary.apply_rotary = cpu_safe_apply_rotary
    _ROTARY_PATCHED = True


def resolve_evo2_checkpoint(model_name: str = "evo2_7b", local_path: str | Path | None = None) -> str | None:
    model_name = _normalize_vendor_evo2_model_name(model_name)
    if local_path is not None:
        return str(Path(local_path).expanduser().resolve())
    default_path = DEFAULT_EVO2_CHECKPOINTS.get(model_name)
    if default_path is not None and default_path.exists():
        return str(default_path)
    return None


def _resolve_target_device(target_device: str | torch.device | None) -> torch.device | None:
    if target_device is None:
        return None
    if isinstance(target_device, torch.device):
        return target_device
    if target_device == "cuda":
        return torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device(target_device)


def _apply_model_dtype(evo2_model: object, dtype: torch.dtype | None) -> object:
    if dtype is not None:
        evo2_model.model = evo2_model.model.to(dtype=dtype)
    return evo2_model


@contextmanager
def _patch_stripedhyena_single_device(target_device: torch.device | None):
    if target_device is None:
        yield
        return

    import vortex.model.model as model_mod

    original_init = model_mod.StripedHyena.__init__
    target_device_str = str(target_device)

    def patched_init(self, config):
        torch.nn.Module.__init__(self)
        if model_mod.HAS_TE:
            model_mod.fixup_te_workspace()

        if config.get("use_fp8_input_projections", False) and not model_mod.HAS_TE:
            raise ImportError(
                "This model requires FP8 input projections (use_fp8_input_projections=True) "
                "which depends on Transformer Engine, but TE is not installed.\n"
                "Only 7b models (8k, 262k, 1m) can run without Transformer Engine."
            )

        self.config = config
        self.print_activations = config.get("print_activations", False)

        if self.print_activations:
            model_mod.enable_activations_logging()
        self.logger = model_mod.logging.getLogger(self.__class__.__name__)

        self.ground_truth_activations_path = config.get("ground_truth_activations_path", None)
        self.logger.info(f"Initializing StripedHyena with config: {config}")

        alloc_device = target_device_str if target_device.type != "cpu" else "cpu"

        def cuda_ctx():
            return model_mod.torch.cuda.device(target_device_str) if target_device.type == "cuda" else nullcontext()

        with model_mod.torch.device(alloc_device):
            self.embedding_layer = model_mod.VocabParallelEmbedding(config)

        if config.get("use_flashfft", "True"):
            try:
                from flashfftconv import FlashFFTConv

                self.flash_fft = FlashFFTConv(config.seqlen, dtype=model_mod.torch.bfloat16)
            except ImportError:
                self.flash_fft = None
        else:
            self.flash_fft = None

        if not self.config.get("evo2_style_activations", False):
            self.logger.warning(
                "⚠️  Not using Evo2 style activations  ⚠️\n"
                "⚠️ Set 'evo2_style_activations: True' in config if you are using Evo 2 checkpoints ⚠️"
            )
        self.logger.info(f"Initializing {config.num_layers} blocks...")
        self.blocks = model_mod.nn.ModuleList()
        self.block_idx_to_device = {}

        self.logger.info(f"Placing all {config.num_layers} layers on {target_device_str}")

        for layer_idx in model_mod.tqdm(range(config.num_layers)):
            with model_mod.torch.device(alloc_device):
                with cuda_ctx():
                    block = model_mod.get_block(config, layer_idx, flash_fft=self.flash_fft)
                    model_mod.move_to_device(block, alloc_device)

            self.blocks.append(block)
            self.block_idx_to_device[layer_idx] = alloc_device
            self.logger.info(f"Assigned {layer_idx=} to device='{alloc_device}'")
            self.logger.info(
                f"Parameter count for block {layer_idx}: {sum(p.numel() for p in self.blocks[-1].parameters())}"
            )

        with model_mod.torch.device(alloc_device):
            with cuda_ctx():
                self.norm = model_mod.RMSNorm(config) if config.get("final_norm", True) else None
                if config.tie_embeddings:
                    self.unembed = model_mod.Lambda(self.embedding_layer.unembed)
                else:
                    self.unembed = model_mod.VocabParallelUnembedding(config)

        self.logger.info("Initialized model")

    model_mod.StripedHyena.__init__ = patched_init
    try:
        yield
    finally:
        model_mod.StripedHyena.__init__ = original_init


class Evo2:
    def __new__(
        cls,
        model_name: str = "evo2_7b",
        local_path: str | Path | None = None,
        target_device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        *args,
        **kwargs,
    ):
        vendor_evo2 = _get_vendor_evo2_class()
        model_name = _normalize_vendor_evo2_model_name(model_name)
        _patch_rotary_for_cpu()
        evo2_kwargs = {
            "model_name": model_name,
            "local_path": resolve_evo2_checkpoint(model_name, local_path),
            **kwargs,
        }
        resolved_target_device = _resolve_target_device(target_device)
        with _patch_stripedhyena_single_device(resolved_target_device):
            if torch.cuda.is_available():
                evo2 = vendor_evo2(*args, **evo2_kwargs)
            else:
                with patch("torch.cuda.device", new=lambda *_args, **_kwargs: nullcontext()):
                    evo2 = vendor_evo2(*args, **evo2_kwargs)

        return _apply_model_dtype(evo2, dtype)


def load_evo2(
    model_name: str = "evo2_7b",
    local_path: str | Path | None = None,
    target_device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Evo2:
    return Evo2(model_name=model_name, local_path=local_path, target_device=target_device, dtype=dtype)
