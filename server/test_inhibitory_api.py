import asyncio
from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks, Response


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
