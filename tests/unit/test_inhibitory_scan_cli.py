import json
from types import SimpleNamespace

import torch

from llamascopium.circuits.global_weights import InhibitoryConnection
from llamascopium.cli import run


def test_dataset_scan_exports_ranked_single_target_result(monkeypatch, tmp_path):
    class FakeModel:
        cfg = SimpleNamespace(n_ctx=3)
        lm_cfg = SimpleNamespace(prepend_bos=True)
        device = torch.device("cpu")

        def eval(self):
            return self

        def to_tokens(self, text, **kwargs):
            assert kwargs == {"prepend_bos": True, "truncate": True}
            return torch.tensor([[1, 2, 3]])

    class FakeSAE:
        def __init__(self, name):
            hooks = {
                "source": "blocks.0.hook_mlp_out",
                "target": "blocks.1.hook_mlp_out",
                "later-matryoshka": "blocks.2.hook_resid_post",
            }
            self.cfg = SimpleNamespace(d_sae=4, hook_point_out=hooks[name])

        def eval(self):
            return self

    class FakeClient:
        def __init__(self, cfg):
            pass

        def get_sae_set(self, name):
            return SimpleNamespace(sae_series="series", sae_names=["source", "target", "later-matryoshka"])

        def get_sae(self, name, series):
            return SimpleNamespace(cfg=FakeSAE(name).cfg)

        def get_sae_model_name(self, name, series):
            return "model"

        def get_model_cfg(self, name):
            return SimpleNamespace(device="cpu", dtype=torch.float32)

        def get_dataset_cfg(self, name):
            return SimpleNamespace()

        def get_sae_path(self, name, series):
            assert name != "later-matryoshka"
            return name

    monkeypatch.setattr(run, "MongoClient", FakeClient)
    monkeypatch.setattr(run, "TransformerLensLanguageModel", FakeModel)
    monkeypatch.setattr(run, "SparseAutoEncoder", FakeSAE)
    monkeypatch.setattr(run, "LowRankSparseAttention", FakeSAE)
    monkeypatch.setattr(run, "SparseDictionary", SimpleNamespace(from_pretrained=lambda path, **kwargs: FakeSAE(path)))
    monkeypatch.setattr(run, "load_model", lambda cfg: FakeModel())
    monkeypatch.setattr(
        run,
        "load_dataset_shard",
        lambda *args, **kwargs: iter([{"text": "hello"}, {"tokens": [4, 5, 6, 7]}, {"text": "later"}]),
    )

    def compute(model, saes, inputs, targets):
        assert len(saes) == 2
        samples = list(inputs)
        assert [sample.tolist() for sample in samples] == [[[1, 2, 3]], [[4, 5, 6]]]
        assert targets == [("blocks.1.hook_mlp_out", 3)]
        edge = InhibitoryConnection(
            "blocks.0.hook_mlp_out", 2, "blocks.1.hook_mlp_out", 3, 2.0, 1.0, 1.0, -2.0, 0.0, 1.0
        )
        return SimpleNamespace(num_samples=2, n_total=6, topk=lambda *args, **kwargs: [edge])

    monkeypatch.setattr(run, "compute_inhibitory_global_weights", compute)
    output = tmp_path / "result.json"
    run.scan_inhibitory_global_weights(
        sae_set="set",
        target_sae="target",
        feature_id=3,
        dataset="dataset",
        output=output,
        max_samples=2,
        top_k=10,
        shard_idx=0,
        n_shards=1,
        normalized=False,
        device="cpu",
        series="series",
        mongo_uri="mongodb://localhost:27017",
        mongo_db="test",
    )
    saved = json.loads(output.read_text())
    assert saved["target"] == {"saeName": "target", "featureIndex": 3}
    assert saved["numSamples"] == 2
    assert saved["numPositions"] == 6
    assert saved["shardIndex"] == 0
    assert saved["numShards"] == 1
    assert saved["maxSamples"] == 2
    assert saved["upstream"][0]["sourceSaeName"] == "source"
    assert saved["upstream"][0]["score"] == 2.0
    assert saved["downstream"] == []
