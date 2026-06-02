import torch

from llamascopium.backend.language_model import Evo2LanguageModel, LanguageModelConfig


def test_evo2_preprocess_raw_data_random_crop_is_deterministic() -> None:
    model = Evo2LanguageModel.__new__(Evo2LanguageModel)
    model.cfg = LanguageModelConfig(
        model_name="evo2_7b",
        backend="evo2",
        max_length=4,
        random_crop_to_max_length=True,
        random_crop_seed=7,
    )

    raw = {
        "text": ["ABCDEFGHIJ", "ABCDEFGHIJ"],
        "meta": [
            {"context_idx": 3, "shard_idx": 1},
            {"context_idx": 3, "shard_idx": 1},
        ],
    }

    processed_a = model.preprocess_raw_data(raw)
    processed_b = model.preprocess_raw_data(raw)

    assert processed_a["text"] == processed_b["text"]
    assert len(processed_a["text"][0]) == 4
    assert processed_a["text"][0] == processed_a["text"][1]


def test_evo2_preprocess_raw_data_skips_short_sequences() -> None:
    model = Evo2LanguageModel.__new__(Evo2LanguageModel)
    model.cfg = LanguageModelConfig(
        model_name="evo2_7b",
        backend="evo2",
        max_length=128_000,
        random_crop_to_max_length=True,
        random_crop_seed=42,
    )

    raw = {"text": ["ACGTACGT"], "meta": [{"context_idx": 0, "shard_idx": 0}]}

    processed = model.preprocess_raw_data(raw)

    assert processed["text"] == raw["text"]


def test_evo2_move_activation_to_capture_device_cpu() -> None:
    model = Evo2LanguageModel.__new__(Evo2LanguageModel)
    model.activation_device = torch.device("cpu")

    tensor = model._move_activation_to_capture_device(torch.ones(2, 3))

    assert tensor.device.type == "cpu"
