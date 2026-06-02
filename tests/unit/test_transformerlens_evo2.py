import torch

from llamascopium.backend.language_model import LanguageModelConfig, TransformerLensLanguageModel
from llamascopium.resource_loaders import load_model


def test_transformerlens_preprocess_raw_data_random_crop_is_deterministic() -> None:
    model = TransformerLensLanguageModel.__new__(TransformerLensLanguageModel)
    model.lm_cfg = LanguageModelConfig(
        model_name="evo2_7b",
        backend="transformer_lens",
        max_length=4,
        random_crop_to_max_length=True,
        random_crop_seed=9,
    )

    raw = {
        "text": ["ABCDEFGHIJ", "ABCDEFGHIJ"],
        "meta": [
            {"context_idx": 5, "shard_idx": 2},
            {"context_idx": 5, "shard_idx": 2},
        ],
    }

    processed_a = model.preprocess_raw_data(raw)
    processed_b = model.preprocess_raw_data(raw)

    assert processed_a["text"] == processed_b["text"]
    assert len(processed_a["text"][0]) == 4
    assert processed_a["text"][0] == processed_a["text"][1]


def test_load_model_routes_evo2_transformerlens_backend(monkeypatch) -> None:
    cfg = LanguageModelConfig(model_name="evo2_7b", backend="transformer_lens", dtype=torch.bfloat16)
    sentinel = object()

    def fake_from_pretrained_evo2(inner_cfg, device_mesh=None):
        assert inner_cfg is cfg
        assert device_mesh is None
        return sentinel

    monkeypatch.setattr(TransformerLensLanguageModel, "from_pretrained_evo2", staticmethod(fake_from_pretrained_evo2))

    model = load_model(cfg)

    assert model is sentinel


def test_transformerlens_evo2_loader_passes_local_path(monkeypatch) -> None:
    cfg = LanguageModelConfig(model_name="arcinstitute/evo2_7b", backend="transformer_lens", dtype=torch.bfloat16)
    captured = {}

    class _DummyHooked:
        pass

    def fake_from_pretrained(model_name, **kwargs):
        captured["model_name"] = model_name
        captured.update(kwargs)
        return _DummyHooked()

    def fake_from_hooked_transformer(model, device_mesh=None, **overrides):
        return {"model": model, "device_mesh": device_mesh, "overrides": overrides}

    monkeypatch.setattr("llamascopium.backend.language_model.resolve_evo2_checkpoint", lambda _name: "/tmp/evo2_7b.pt")
    monkeypatch.setattr("llamascopium.backend.language_model.HookedTransformer.from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(
        TransformerLensLanguageModel,
        "from_hooked_transformer",
        staticmethod(fake_from_hooked_transformer),
    )

    result = TransformerLensLanguageModel.from_pretrained_evo2(cfg)

    assert captured["model_name"] == "arcinstitute/evo2_7b"
    assert captured["local_path"] == "/tmp/evo2_7b.pt"
    assert captured["local_files_only"] is True
    assert isinstance(result["model"], _DummyHooked)


def test_transformerlens_evo2_loader_enables_offline_env(monkeypatch) -> None:
    cfg = LanguageModelConfig(model_name="arcinstitute/evo2_7b", backend="transformer_lens", dtype=torch.bfloat16)
    captured = {}

    class _DummyHooked:
        pass

    def fake_from_pretrained(model_name, **kwargs):
        import os

        captured["model_name"] = model_name
        captured["hf_hub_offline"] = os.environ.get("HF_HUB_OFFLINE")
        captured["transformers_offline"] = os.environ.get("TRANSFORMERS_OFFLINE")
        return _DummyHooked()

    def fake_from_hooked_transformer(model, device_mesh=None, **overrides):
        return model

    monkeypatch.setattr("llamascopium.backend.language_model.resolve_evo2_checkpoint", lambda _name: "/tmp/evo2_7b.pt")
    monkeypatch.setattr("llamascopium.backend.language_model.HookedTransformer.from_pretrained_no_processing", fake_from_pretrained)
    monkeypatch.setattr(
        TransformerLensLanguageModel,
        "from_hooked_transformer",
        staticmethod(fake_from_hooked_transformer),
    )

    model = TransformerLensLanguageModel.from_pretrained_evo2(cfg)

    assert isinstance(model, _DummyHooked)
    assert captured["hf_hub_offline"] == "1"
    assert captured["transformers_offline"] == "1"
