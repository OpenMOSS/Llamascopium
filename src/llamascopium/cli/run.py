import json
import re
from pathlib import Path
from typing import Annotated

import torch
import typer

from llamascopium.backend.language_model import TransformerLensLanguageModel
from llamascopium.circuits.global_weights import compute_inhibitory_global_weights, search_inhibitory_atlas
from llamascopium.database import MongoClient, MongoDBConfig
from llamascopium.models.lorsa import LowRankSparseAttention
from llamascopium.models.sae import SparseAutoEncoder
from llamascopium.models.sparse_dictionary import SparseDictionary
from llamascopium.resource_loaders import load_dataset_shard, load_model
from llamascopium.runners import (
    AnalyzeSAESettings,
    GenerateActivationsSettings,
    TrainSAESettings,
)
from llamascopium.runners import (
    analyze_sae as run_analyze,
)
from llamascopium.runners import (
    generate_activations as run_generate,
)
from llamascopium.runners import (
    train_sae as run_train,
)

from .common import (
    DEFAULT_MONGO_DB,
    DEFAULT_MONGO_URI,
    DEFAULT_SAE_SERIES,
    MongoDBOption,
    MongoURIOption,
    SAESeriesOption,
)
from .utils import load_config

app = typer.Typer()


def _hook_order(hook_point: str) -> tuple[int, int]:
    match = re.fullmatch(r"blocks\.(\d+)\.hook_(attn|mlp)_out", hook_point)
    if match is None:
        raise typer.BadParameter(f"Unsupported SAE output hook: {hook_point}")
    return int(match.group(1)), 0 if match.group(2) == "attn" else 1


@app.command("generate")
def generate_activations(
    config: Annotated[Path, typer.Argument(help="Path to GenerateActivationsSettings configuration file.")],
) -> None:
    """Generate activations from a language model."""
    cfg = load_config(config)
    settings = GenerateActivationsSettings.model_validate(cfg)
    run_generate(settings)


@app.command("train")
def train(
    config: Annotated[Path, typer.Argument(help="Path to TrainSAESettings configuration file.")],
) -> None:
    """Train a Sparse Autoencoder."""
    cfg = load_config(config)
    settings = TrainSAESettings.model_validate(cfg)
    run_train(settings)


@app.command("analyze")
def analyze(
    config: Annotated[Path, typer.Argument(help="Path to AnalyzeSAESettings configuration file.")],
) -> None:
    """Analyze a trained Sparse Autoencoder."""
    cfg = load_config(config)
    settings = AnalyzeSAESettings.model_validate(cfg)
    run_analyze(settings)


@app.command("inhibitory-global-weights")
def scan_inhibitory_global_weights(
    sae_set: Annotated[str, typer.Option("--sae-set", help="Registered SAE set containing the target and sources.")],
    target_sae: Annotated[str, typer.Option("--target-sae", help="Registered SAE name containing the target feature.")],
    feature_id: Annotated[int, typer.Option("--feature-id", min=0, help="Target feature index in target SAE.")],
    dataset: Annotated[str, typer.Option("--dataset", help="Registered dataset name.")],
    output: Annotated[Path, typer.Option("--output", help="JSON result file for later visualization.")],
    max_samples: Annotated[int, typer.Option("--max-samples", min=0, help="Maximum valid rows; 0 scans the full shard.")] = 0,
    top_k: Annotated[int, typer.Option("--top-k", min=1)] = 50,
    depth: Annotated[int, typer.Option("--depth", min=1, max=3, help="Number of upstream expansion levels.")] = 2,
    expansion_width: Annotated[int, typer.Option("--expansion-width", min=1, max=5)] = 3,
    shard_idx: Annotated[int, typer.Option("--shard-idx", min=0)] = 0,
    n_shards: Annotated[int, typer.Option("--n-shards", min=1)] = 1,
    normalized: Annotated[bool, typer.Option("--normalized")] = False,
    device: Annotated[str, typer.Option("--device")] = "cuda",
    series: SAESeriesOption = DEFAULT_SAE_SERIES,
    mongo_uri: MongoURIOption = DEFAULT_MONGO_URI,
    mongo_db: MongoDBOption = DEFAULT_MONGO_DB,
) -> None:
    """Scan a dataset for upstream observational suppression of one feature."""
    if shard_idx >= n_shards:
        raise typer.BadParameter("shard-idx must be less than n-shards")
    if device == "cuda" and not torch.cuda.is_available():
        raise typer.BadParameter("CUDA is unavailable; run this command on a GPU or set --device cpu")
    client = MongoClient(MongoDBConfig(mongo_uri=mongo_uri, mongo_db=mongo_db))
    sae_set_record = client.get_sae_set(sae_set)
    if sae_set_record is None or sae_set_record.sae_series != series or target_sae not in sae_set_record.sae_names:
        raise typer.BadParameter("Target SAE is not in the selected SAE set and series")
    model_name = client.get_sae_model_name(target_sae, series)
    model_cfg = client.get_model_cfg(model_name) if model_name else None
    dataset_cfg = client.get_dataset_cfg(dataset)
    if model_cfg is None or dataset_cfg is None:
        raise typer.BadParameter("The target model or dataset is not registered")
    model_cfg.device = device
    model_cfg.dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = load_model(model_cfg)
    if not isinstance(model, TransformerLensLanguageModel):
        raise typer.BadParameter("Inhibitory global weights require a TransformerLens model")
    model.eval()

    target_record = client.get_sae(target_sae, series)
    if target_record is None:
        raise typer.BadParameter(f"SAE {target_sae} is not registered")
    target_order = _hook_order(target_record.cfg.hook_point_out)
    saes = []
    selected_names = []
    for name in sae_set_record.sae_names:
        record = client.get_sae(name, series)
        if record is None:
            raise typer.BadParameter(f"SAE {name} is not registered")
        hook_point = record.cfg.hook_point_out
        if name != target_sae and hook_point.endswith(".hook_resid_post"):
            continue
        if name != target_sae and _hook_order(hook_point) >= target_order:
            continue
        if client.get_sae_model_name(name, series) != model_name:
            raise typer.BadParameter(f"SAE {name} belongs to a different model")
        path = client.get_sae_path(name, series)
        if path is None:
            raise typer.BadParameter(f"SAE {name} is not registered")
        sae = SparseDictionary.from_pretrained(path, device=device, dtype=model_cfg.dtype)
        if not isinstance(sae, SparseAutoEncoder | LowRankSparseAttention):
            raise typer.BadParameter(f"SAE {name} is not a supported transcoder or Lorsa")
        sae.eval()
        saes.append(sae)
        selected_names.append(name)
    selected = saes[selected_names.index(target_sae)]
    if feature_id >= selected.cfg.d_sae:
        raise typer.BadParameter(f"Feature {feature_id} is outside {target_sae}")
    target = (selected.cfg.hook_point_out, feature_id)
    def inputs():
        scanned = 0
        for row in load_dataset_shard(dataset_cfg, shard_idx=shard_idx, n_shards=n_shards):
            if max_samples and scanned >= max_samples:
                break
            if isinstance(row.get("text"), str) and row["text"].strip():
                sample = model.to_tokens(row["text"], prepend_bos=model.lm_cfg.prepend_bos, truncate=True)
            elif row.get("tokens") is not None:
                tokens = torch.as_tensor(row["tokens"], dtype=torch.long)
                if tokens.ndim != 1:
                    raise ValueError("Dataset token rows must be one-dimensional")
                sample = tokens[: model.cfg.n_ctx].unsqueeze(0).to(model.device)
            else:
                continue
            if sample.shape[1] == 0:
                continue
            scanned += 1
            if scanned % 100 == 0:
                typer.echo(f"Scanned {scanned} samples")
            yield sample

    atlas, result = search_inhibitory_atlas(
        target,
        lambda targets: compute_inhibitory_global_weights(model, saes, inputs(), targets),
        depth=depth,
        top_k=top_k,
        expansion_width=expansion_width,
        normalized=normalized,
    )
    if result.num_samples == 0:
        raise typer.BadParameter("Dataset contains no nonempty text or token rows")
    names_by_hook = {sae.cfg.hook_point_out: name for name, sae in zip(selected_names, saes)}
    edges = [edge for edge in atlas.links.values() if (edge.target, edge.target_feature) == target]

    def serialize(edge):
        return {
            "source": edge.source,
            "sourceSaeName": names_by_hook[edge.source],
            "sourceFeature": edge.source_feature,
            "target": edge.target,
            "targetSaeName": names_by_hook[edge.target],
            "targetFeature": edge.target_feature,
            "kind": edge.kind,
            "score": edge.score,
            "normalizedInhibitoryScore": edge.normalized_inhibitory_score,
            "delta": edge.delta,
            "virtualWeight": edge.virtual_weight,
            "muOn": edge.mu_on,
            "muOff": edge.mu_off,
        }

    payload = {
        "schemaVersion": 1,
        "mode": "inhibitory",
        "datasetName": dataset,
        "shardIndex": shard_idx,
        "numShards": n_shards,
        "maxSamples": max_samples,
        "saeSetName": sae_set,
        "saeSeries": series,
        "modelName": model_name,
        "target": {"saeName": target_sae, "featureIndex": feature_id},
        "numSamples": result.num_samples,
        "numPositions": result.n_total,
        "depth": depth,
        "upstream": [serialize(edge) for edge in edges],
        "downstream": [],
        "connections": [serialize(edge) for edge in atlas.links.values()],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    typer.echo(f"Saved {len(edges)} upstream features from {result.num_samples} samples to {output}")
    for rank, edge in enumerate(edges[:10], start=1):
        value = edge.normalized_inhibitory_score if normalized else edge.score
        typer.echo(f"{rank:>2}. {names_by_hook[edge.source]} #{edge.source_feature}: {value:.6g}")
