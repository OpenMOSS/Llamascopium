import torch

from llamascopium.backend.evo2 import _apply_model_dtype


class _FakeEvo2:
    def __init__(self) -> None:
        self.model = torch.nn.Linear(4, 4)


def test_apply_model_dtype_casts_vendor_model() -> None:
    evo2 = _FakeEvo2()

    _apply_model_dtype(evo2, torch.bfloat16)

    assert evo2.model.weight.dtype == torch.bfloat16
    assert evo2.model.bias.dtype == torch.bfloat16


def test_apply_model_dtype_noop_when_none() -> None:
    evo2 = _FakeEvo2()
    original_dtype = evo2.model.weight.dtype

    _apply_model_dtype(evo2, None)

    assert evo2.model.weight.dtype == original_dtype
