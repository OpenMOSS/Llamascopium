import asyncio
from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks, Response

from llamascopium.circuits.global_weights import GlobalConnection, InhibitoryConnection


@pytest.fixture
def circuits(monkeypatch):
    import llamascopium

    monkeypatch.setattr(llamascopium, "MongoClient", lambda cfg: SimpleNamespace())
    from server.routers import circuits as circuits_router

    monkeypatch.setattr(circuits_router, "_inhibitory_jobs", {})
    return circuits_router


def test_inhibitory_job_status_is_scoped_to_its_circuit(monkeypatch, circuits):
    records = {
        "circuit-a": SimpleNamespace(
            sae_series=circuits.sae_series,
            status=circuits.CircuitStatus.COMPLETED,
            sae_set_name="set-a",
        ),
        "circuit-b": SimpleNamespace(
            sae_series=circuits.sae_series,
            status=circuits.CircuitStatus.COMPLETED,
            sae_set_name="set-a",
        ),
    }
    monkeypatch.setattr(circuits.client, "get_circuit", lambda circuit_id: records.get(circuit_id), raising=False)
    monkeypatch.setattr(circuits.client, "get_sae_set", lambda name: SimpleNamespace(sae_names=["sae-a"]), raising=False)
    request = circuits.InhibitoryRequest(sae_name="sae-a", feature_index=3, prompts=["hello"])
    tasks = BackgroundTasks()

    started = circuits.start_inhibitory_job("circuit-a", request, tasks)
    assert started["status"] == "pending"
    assert len(tasks.tasks) == 1
    job_id = started["job_id"]
    assert circuits.get_inhibitory_job("circuit-a", job_id) == {"status": "pending"}
    assert isinstance(circuits.get_inhibitory_job("circuit-b", job_id), Response)

    async def compute(sae_set_name, submitted_request):
        assert sae_set_name == "set-a"
        assert submitted_request is request
        return {"edges": [], "num_samples": 1}

    monkeypatch.setattr(circuits, "_compute_inhibitory_job", compute)
    asyncio.run(circuits._run_inhibitory_job(job_id, "circuit-a", "set-a", request))
    assert circuits.get_inhibitory_job("circuit-a", job_id) == {
        "status": "completed",
        "result": {"edges": [], "num_samples": 1},
    }


def test_inhibitory_request_rejects_empty_prompts(monkeypatch, circuits):
    monkeypatch.setattr(
        circuits.client,
        "get_circuit",
        lambda _: SimpleNamespace(
            sae_series=circuits.sae_series,
            status=circuits.CircuitStatus.COMPLETED,
            sae_set_name="set-a",
        ),
        raising=False,
    )
    request = circuits.InhibitoryRequest(sae_name="sae-a", feature_index=3, prompts=[" "])
    response = circuits.start_inhibitory_job("circuit-a", request, BackgroundTasks())
    assert isinstance(response, Response)
    assert response.status_code == 400


def test_global_weight_job_dispatches_both_modes_and_keeps_edge_semantics(monkeypatch, circuits):
    class Model:
        pass

    source = SimpleNamespace(cfg=SimpleNamespace(d_sae=4, hook_point_out="blocks.0.hook_mlp_out"))
    target = SimpleNamespace(cfg=SimpleNamespace(d_sae=5, hook_point_out="blocks.1.hook_attn_out"))
    monkeypatch.setattr(circuits, "TransformerLensLanguageModel", Model)
    monkeypatch.setattr(
        circuits.client,
        "get_sae_set",
        lambda name: SimpleNamespace(sae_names=["source-sae", "target-sae"], sae_series=circuits.sae_series),
        raising=False,
    )
    monkeypatch.setattr(circuits.client, "get_sae_model_name", lambda *args: "model", raising=False)
    monkeypatch.setattr(circuits, "get_sae", lambda name, device_mesh: source if name == "source-sae" else target)
    monkeypatch.setattr(circuits, "get_model", lambda name, device_mesh: Model())
    monkeypatch.setattr(circuits, "SparseAutoEncoder", SimpleNamespace)
    monkeypatch.setattr(circuits, "LowRankSparseAttention", SimpleNamespace)

    global_edge = GlobalConnection(
        source="blocks.0.hook_mlp_out",
        source_feature=2,
        target="blocks.1.hook_attn_out",
        target_feature=3,
        weight=-0.5,
    )
    inhibitory_edge = InhibitoryConnection(
        source="blocks.0.hook_mlp_out",
        source_feature=2,
        target="blocks.1.hook_attn_out",
        target_feature=3,
        score=2.0,
        normalized_inhibitory_score=1.0,
        delta=1.0,
        virtual_weight=-2.0,
        mu_on=0.0,
        mu_off=1.0,
    )
    monkeypatch.setattr(
        circuits,
        "compute_global_weights",
        lambda *args: SimpleNamespace(num_samples=1, topk=lambda *args, **kwargs: [global_edge], downstream=lambda *args, **kwargs: [global_edge]),
    )
    monkeypatch.setattr(
        circuits,
        "compute_inhibitory_global_weights",
        lambda *args: SimpleNamespace(num_samples=1, n_total=3, topk=lambda *args, **kwargs: [inhibitory_edge]),
    )

    global_request = circuits.GlobalWeightRequest(sae_name="target-sae", feature_index=3, prompts=["hello"])
    global_result = circuits._compute_global_weight_job.__wrapped__("set-a", global_request)
    assert global_result["upstream"][0]["weight"] == -0.5
    assert global_result["downstream"][0]["target_sae_name"] == "target-sae"
    assert global_result["upstream"][0]["kind"] == "global"

    inhibitory_request = global_request.model_copy(update={"mode": "inhibitory"})
    inhibitory_result = circuits._compute_global_weight_job.__wrapped__("set-a", inhibitory_request)
    assert inhibitory_result["upstream"][0]["kind"] == "inhibitory"
    assert inhibitory_result["upstream"][0]["score"] == 2.0
    assert inhibitory_result["downstream"] == []


def test_global_weight_job_status_is_scoped_to_its_circuit(monkeypatch, circuits):
    monkeypatch.setattr(circuits, "_global_weight_jobs", {})
    records = {
        circuit_id: SimpleNamespace(
            sae_series=circuits.sae_series,
            status=circuits.CircuitStatus.COMPLETED,
            sae_set_name="set-a",
        )
        for circuit_id in ("circuit-a", "circuit-b")
    }
    monkeypatch.setattr(circuits.client, "get_circuit", lambda circuit_id: records.get(circuit_id), raising=False)
    monkeypatch.setattr(circuits.client, "get_sae_set", lambda name: SimpleNamespace(sae_names=["sae-a"]), raising=False)
    request = circuits.GlobalWeightRequest(sae_name="sae-a", feature_index=3, prompts=["hello"])
    tasks = BackgroundTasks()
    started = circuits.start_global_weight_job("circuit-a", request, tasks)
    assert started["status"] == "pending"
    assert len(tasks.tasks) == 1
    job_id = started["job_id"]
    assert circuits.get_global_weight_job("circuit-a", job_id) == {"status": "pending"}
    assert isinstance(circuits.get_global_weight_job("circuit-b", job_id), Response)
