import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from llamascopium.circuits.indexed_tensor import NodeDimension, NodeInfo
from llamascopium.testing import distributed_test


@distributed_test(nproc_per_node=2, backend="gloo")
def test_node_dimension_subtracts_replicated_dtensor():
    device_mesh = init_device_mesh(
        "cpu",
        (dist.get_world_size(),),
        mesh_dim_names=("model",),
    )
    full = NodeDimension.from_node_infos(
        [NodeInfo(key="matry", indices=torch.tensor([[0, 1], [0, 2], [0, 3]]))],
        device="cpu",
        device_mesh=device_mesh,
    )
    remove = NodeDimension.from_node_infos(
        [NodeInfo(key="matry", indices=torch.tensor([[0, 2]]))],
        device="cpu",
        device_mesh=device_mesh,
    )

    result = full - remove

    assert torch.equal(
        result.node_mappings["matry"].indices,
        torch.tensor([[0, 1], [0, 3]]),
    )
