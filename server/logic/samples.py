from functools import lru_cache
from typing import Any, Generator

import numpy as np

from llamascopium import FeatureAnalysisSampling
from server.config import LRU_CACHE_SIZE_SAMPLES, client, sae_series
from server.logic.loaders import get_dataset, get_model


@lru_cache(maxsize=LRU_CACHE_SIZE_SAMPLES)
def cached_extract_samples(
    sae_name: str,
    sae_series: str,
    feature_index: int,
    sampling_name: str,
    start: int | None = None,
    end: int | None = None,
    visible_range: int | None = None,
) -> list[dict[str, Any]]:
    """LRU-cached ``extract_samples``, keyed by feature identifiers.

    Fetches the feature from the database, locates the matching sampling by
    *sampling_name*, and delegates to :func:`extract_samples`.
    """
    feature = client.get_feature(sae_name=sae_name, sae_series=sae_series, index=feature_index)
    if feature is None:
        return []
    for analysis in feature.analyses:
        sampling = next((s for s in analysis.samplings if s.name == sampling_name), None)
        if sampling is not None:
            return extract_samples(sampling, start, end, visible_range)
    return []


def extract_samples(
    sampling: FeatureAnalysisSampling,
    start: int | None = None,
    end: int | None = None,
    visible_range: int | None = None,
) -> list[dict[str, Any]]:
    def process_sample(
        *,
        sparse_feature_acts: tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None],
        context_idx: int,
        dataset_name: str,
        model_name: str,
        shard_idx: int | None = None,
        n_shards: int | None = None,
    ):
        model = get_model(name=model_name)
        data = get_dataset(name=dataset_name, shard_idx=shard_idx, n_shards=n_shards)[context_idx]

        origins = model.trace({k: [v] for k, v in data.items()})[0]

        (
            feature_acts_indices,
            feature_acts_values,
            z_pattern_indices,
            z_pattern_values,
        ) = sparse_feature_acts

        assert origins is not None and feature_acts_indices is not None and feature_acts_values is not None, (
            "Origins and feature acts must not be None"
        )

        # Stored sample indices are over analyzable tokens, while trace origins
        # include the leading BOS token position. Return trace-aligned indices
        # so the frontend can index origins directly.
        token_index_offset = 1
        token_offset = 0
        if visible_range is not None:  # Drop tokens before and after the highest activating token
            if len(feature_acts_indices) == 0:
                max_feature_act_index = 0
            else:
                max_feature_act_index = int(feature_acts_indices[np.argmax(feature_acts_values).item()].item())

            feature_acts_mask = np.logical_and(
                feature_acts_indices > max_feature_act_index - visible_range,
                feature_acts_indices < max_feature_act_index + visible_range,
            )
            feature_acts_indices = feature_acts_indices[feature_acts_mask]
            feature_acts_values = feature_acts_values[feature_acts_mask]

            if z_pattern_indices is not None and z_pattern_values is not None:
                z_pattern_mask = np.logical_and(
                    z_pattern_indices > max_feature_act_index - visible_range,
                    z_pattern_indices < max_feature_act_index + visible_range,
                ).all(axis=0)
                z_pattern_indices = z_pattern_indices[:, z_pattern_mask]
                z_pattern_values = z_pattern_values[z_pattern_mask]

            token_offset = max(0, max_feature_act_index - visible_range)

            origins = origins[token_offset : max_feature_act_index + visible_range]

        if token_index_offset:
            feature_acts_indices = feature_acts_indices + token_index_offset
            if z_pattern_indices is not None:
                z_pattern_indices = z_pattern_indices + token_index_offset

        text_offset = None
        if "text" in data:
            text_ranges = [origin["range"] for origin in origins if origin is not None and origin["key"] == "text"]
            if text_ranges:
                max_text_origin = max(text_ranges, key=lambda x: x[1])
                data["text"] = data["text"][: max_text_origin[1]]
                if visible_range is not None:
                    text_offset = min(text_ranges, key=lambda x: x[0])[0]
                    data["text"] = data["text"][text_offset:]

        return {
            **data,
            "token_offset": token_offset,
            "text_offset": text_offset,
            "origins": origins,
            "feature_acts_indices": feature_acts_indices,
            "feature_acts_values": feature_acts_values,
            "z_pattern_indices": z_pattern_indices,
            "z_pattern_values": z_pattern_values,
        }

    def index_select(
        indices: np.ndarray,
        values: np.ndarray,
        i: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select i-th sample from sparse tensor indices and values."""
        mask = indices[0] == i
        return indices[1:, mask], values[mask]

    def process_sparse_feature_acts(
        feature_acts_indices: np.ndarray,
        feature_acts_values: np.ndarray,
        z_pattern_indices: np.ndarray | None,
        z_pattern_values: np.ndarray | None,
        start: int,
        end: int,
    ) -> Generator[tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None], Any, None]:
        for i in range(start, end):
            feature_acts_indices_i, feature_acts_values_i = index_select(feature_acts_indices, feature_acts_values, i)
            if z_pattern_indices is not None and z_pattern_values is not None:
                z_pattern_indices_i, z_pattern_values_i = index_select(z_pattern_indices, z_pattern_values, i)
            else:
                z_pattern_indices_i, z_pattern_values_i = None, None
            yield feature_acts_indices_i[0], feature_acts_values_i, z_pattern_indices_i, z_pattern_values_i

    start = start if start is not None else 0
    end = end if end is not None else len(sampling.context_idx)

    return [
        process_sample(
            sparse_feature_acts=sparse_feature_acts,
            context_idx=context_idx,
            dataset_name=dataset_name,
            model_name=model_name,
            shard_idx=shard_idx,
            n_shards=n_shards,
        )
        for sparse_feature_acts, context_idx, dataset_name, model_name, shard_idx, n_shards in zip(
            process_sparse_feature_acts(
                sampling.feature_acts_indices,
                sampling.feature_acts_values,
                sampling.z_pattern_indices,
                sampling.z_pattern_values,
                start,
                end,
            ),
            sampling.context_idx[start:end],
            sampling.dataset_name[start:end],
            sampling.model_name[start:end],
            sampling.shard_idx[start:end] if sampling.shard_idx is not None else [0] * (end - start),
            sampling.n_shards[start:end] if sampling.n_shards is not None else [1] * (end - start),
        )
    ]


def list_feature_data(
    sae_name: str,
    indices: list[int],
    with_samplings: bool = True,
    sampling_size: int = 1,
    sampling_visible_range: int = 10,
    with_logits: bool = True,
) -> dict[tuple[str, int], dict[str, Any]]:
    """List features and (optionally) their associated samples."""
    features = client.list_features(
        sae_name=sae_name, sae_series=sae_series, indices=indices, with_samplings=with_samplings
    )

    features_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for feature in features:
        analysis = next(
            (a for a in feature.analyses if a.name == "default"),
            None,
        )
        if analysis is None:
            analysis = next((a for a in feature.analyses), None)

        if analysis is None:
            continue

        data = {
            "feature_index": feature.index,
            "analysis_name": analysis.name,
            "interpretation": feature.interpretation,
            "dictionary_name": feature.sae_name,
            "act_times": analysis.act_times,
            "max_feature_act": analysis.max_feature_acts,
            "n_analyzed_tokens": analysis.n_analyzed_tokens,
        }

        if with_logits:
            data["logits"] = feature.logits

        if with_samplings:
            sampling = next(
                (s for s in analysis.samplings if s.name == "top_activations"),
                None,
            )
            samples = (
                cached_extract_samples(
                    sae_name,
                    sae_series,
                    feature.index,
                    sampling.name,
                    0,
                    sampling_size,
                    visible_range=sampling_visible_range,
                )
                if sampling is not None
                else []
            )
            data["samples"] = samples

        features_by_key[(sae_name, feature.index)] = data

    return features_by_key
