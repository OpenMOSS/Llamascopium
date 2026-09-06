from collections.abc import Callable

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.optim import Adam, Optimizer

from llamascopium.optim import SparseAdam
from llamascopium.trainer import Trainer


@pytest.mark.parametrize(
    "optimizer_factory",
    [
        pytest.param(lambda params: Adam(params, lr=1e-3), id="adam"),
        pytest.param(lambda params: SparseAdam(params, lr=1e-3), id="sparse-adam"),
    ],
)
def test_distributed_optimizer_checkpoint_restores_lazy_state(
    optimizer_factory: Callable[[list[torch.nn.Parameter]], Optimizer],
    tmp_path,
) -> None:
    source_parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    source_optimizer = optimizer_factory([source_parameter])
    source_parameter.grad = torch.tensor([0.25, -0.5])
    source_optimizer.step()
    source_state = source_optimizer.state_dict()

    checkpoint_path = tmp_path / "optimizer.dcp"
    dcp.save(source_state, storage_writer=FileSystemWriter(checkpoint_path))

    target_parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    target_optimizer = optimizer_factory([target_parameter])
    assert not target_optimizer.state

    Trainer._initialize_optimizer_state_for_checkpoint_load(target_optimizer)
    target_optimizer.param_groups[0]["lr"] = 0.5
    target_state = target_optimizer.state_dict()
    Trainer._load_distributed_state_dict(target_state, FileSystemReader(checkpoint_path))
    target_optimizer.load_state_dict(target_state)

    loaded_state = target_optimizer.state_dict()
    assert loaded_state["param_groups"] == source_state["param_groups"]
    source_parameter_state = source_state["state"][0]
    loaded_parameter_state = loaded_state["state"][0]
    assert loaded_parameter_state.keys() == source_parameter_state.keys()
    for key, expected in source_parameter_state.items():
        actual = loaded_parameter_state[key]
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected)
        else:
            assert actual == expected


def test_distributed_scheduler_checkpoint_restores_python_state(tmp_path) -> None:
    source_parameter = torch.nn.Parameter(torch.tensor([1.0]))
    source_optimizer = Adam([source_parameter], lr=1e-3)
    source_scheduler = torch.optim.lr_scheduler.LinearLR(
        source_optimizer,
        start_factor=0.1,
        total_iters=10,
    )
    for _ in range(4):
        source_optimizer.step()
        source_scheduler.step()
    source_state = source_scheduler.state_dict()

    checkpoint_path = tmp_path / "scheduler.dcp"
    dcp.save(source_state, storage_writer=FileSystemWriter(checkpoint_path))

    target_parameter = torch.nn.Parameter(torch.tensor([1.0]))
    target_optimizer = Adam([target_parameter], lr=1e-3)
    target_scheduler = torch.optim.lr_scheduler.LinearLR(
        target_optimizer,
        start_factor=0.1,
        total_iters=10,
    )
    target_state = target_scheduler.state_dict()
    Trainer._load_distributed_state_dict(target_state, FileSystemReader(checkpoint_path))
    target_scheduler.load_state_dict(target_state)

    assert target_scheduler.state_dict() == source_state
