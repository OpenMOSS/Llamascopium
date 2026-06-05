import warnings
from functools import partial
from typing import Any, Mapping, Optional, cast

import torch
import torch.distributed.tensor
from einops import rearrange, repeat
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from tqdm import tqdm

from lm_saes.abstract_sae import AbstractSparseAutoEncoder
from lm_saes.activation.factory import ActivationFactory
from lm_saes.analysis.post_analysis import get_post_analysis_processor
from lm_saes.clt import CrossLayerTranscoder
from lm_saes.config import FeatureAnalyzerConfig
from lm_saes.crosscoder import CrossCoder
from lm_saes.lorsa import LowRankSparseAttention
from lm_saes.utils.discrete import KeyedDiscreteMapper
from lm_saes.utils.distributed import DimMap, masked_fill, to_local
from lm_saes.utils.misc import is_primary_rank
from lm_saes.utils.tensor_dict import concat_dict_of_tensor, sort_dict_of_tensor


class FeatureAnalyzer:
    """Analyzes feature activations from a sparse autoencoder.

    This class processes activation data from a sparse autoencoder to:
    1. Track activation statistics like frequency and magnitude
    2. Sample and store representative activations
    3. Organize results by feature for analysis
    """

    def __init__(
        self,
        cfg: FeatureAnalyzerConfig,
    ):
        """Initialize the feature analyzer.

        Args:
            cfg: Analysis configuration specifying parameters like sample sizes and thresholds
        """
        self.cfg = cfg

    def _process_batch(
        self,
        feature_acts: torch.Tensor,  # [batch_size, context_size, d_sae]
        discrete_meta: Mapping[str, torch.Tensor],
        sample_result: Mapping[str, Mapping[str, torch.Tensor] | None],
        max_feature_acts: torch.Tensor,  # [d_sae]
        device_mesh: DeviceMesh | None = None,
    ) -> Mapping[str, Mapping[str, torch.Tensor] | None]:
        """Process a batch of activations to update sampling results.

        For each subsample type:
        1. Computes sampling weights if enabled
        2. Filters activations based on configured thresholds
        3. Updates running sample collections
        4. Maintains top N samples by activation magnitude

        Args:
            feature_acts: Feature activation values for current batch
            discrete_meta: Metadata tensors like dataset/context IDs
            sample_result: Current sampling results to update
            max_feature_acts: Maximum activation seen so far per feature
            device_mesh: Device mesh to use for distributed operations
        Returns:
            Updated sampling results with new batch incorporated
        """
        # Compute exponential lottery ticket values for sampling if enabled
        elt = feature_acts.clamp(min=0.0).max(dim=1).values

        # Process each subsample type (e.g. top activations)
        for name in self.cfg.subsamples.keys():
            elt_cur = elt.clone()
            # Zero out samples above the subsample threshold
            index = (
                feature_acts.max(dim=1).values > max_feature_acts.unsqueeze(0) * self.cfg.subsamples[name]["proportion"]
            )
            elt_cur = masked_fill(elt_cur, index, -torch.inf)

            sample_result_cur = sample_result[name]

            # Prepare batch data with proper dimensions
            batch_data = {
                "elt": elt_cur,
                "feature_acts": rearrange(
                    feature_acts,
                    "batch_size context_size d_sae -> batch_size d_sae context_size",
                ),
                # **discrete_meta,
                **{
                    k: repeat(
                        v,
                        "batch_size -> batch_size d_sae",
                        d_sae=feature_acts.size(-1),
                    )
                    for k, v in discrete_meta.items()
                },
            }

            # Initialize or update sample collection
            if sample_result_cur is None:
                sample_result_cur = batch_data
            elif (
                sample_result_cur["elt"].size(0) > 0
                and (elt_cur.max(dim=0).values > sample_result_cur["elt"][-1]).any()
            ):
                sample_result_cur = concat_dict_of_tensor(
                    sample_result_cur,
                    batch_data,
                )
            else:  # Skip if all activations are below the threshold
                continue
            
            # print(f'{sample_result_cur = }')
            
            # Sort and keep top N samples
            sample_result_cur = sort_dict_of_tensor(
                sample_result_cur, sort_dim=0, sort_key="elt", descending=True, device_mesh=device_mesh
            )
            sample_result_cur = {
                k: v[
                    : min(self.cfg.subsamples[name]["n_samples"], (sample_result_cur["elt"] != -torch.inf).sum().item())
                ]
                for k, v in sample_result_cur.items()
            }

            # Update main sample result with current batch results out of place
            sample_result = {**sample_result, name: sample_result_cur}

        return sample_result

    def compute_ignore_token_masks(
        self, tokens: torch.Tensor, ignore_token_ids: Optional[list[int]] = None
    ) -> torch.Tensor:
        """Compute ignore token masks for the given tokens.

        Args:
            tokens: The tokens to compute the ignore token masks for
            ignore_token_ids: The token IDs to ignore
        """
        if ignore_token_ids is None:
            warnings.warn(
                "ignore_token_ids are not provided. No tokens (including pad tokens) will be filtered out. If this is intentional, set ignore_token_ids explicitly to an empty list to avoid this warning.",
                UserWarning,
                stacklevel=2,
            )
            ignore_token_ids = []
        mask = torch.ones_like(tokens, dtype=torch.bool)
        for token_id in ignore_token_ids:
            mask &= tokens != token_id
        return mask

    def get_post_analysis_func(self, sae_type: str):
        """Get the post-analysis processor for the given SAE type.

        Args:
            sae_type: The SAE type identifier

        Returns:
            The post-analysis processor instance
        """
        try:
            return get_post_analysis_processor(sae_type)
        except KeyError:
            # Fallback to generic processor if no specific processor is registered
            return get_post_analysis_processor("generic")

    @torch.no_grad()
    def analyze_chunk(
        self,
        activation_factory: ActivationFactory,
        sae: AbstractSparseAutoEncoder,
        device_mesh: DeviceMesh | None = None,
        activation_factory_process_kwargs: dict[str, Any] = {},
    ) -> list[dict[str, Any]]:
        """Analyze feature activations for a chunk of the SAE.

        Processes activation data to:
        1. Track activation statistics
        2. Sample representative activations
        3. Organize results by feature

        Args:
            activation_factory: The activation factory to use
            sae: The sparse autoencoder model
            device_mesh: The device mesh to use
            activation_factory_process_kwargs: Keyword arguments to pass to the activation factory's process method

        Returns:
            List of dictionaries containing per-feature analysis results:
            - Activation counts and maximums
            - Sampled activations with metadata
        """
        activation_stream = activation_factory.process(**activation_factory_process_kwargs)
        n_tokens = n_analyzed_tokens = 0

        # Progress tracking
        pbar = tqdm(
            total=self.cfg.total_analyzing_tokens,
            desc="Analyzing SAE",
            smoothing=0.01,
            disable=not is_primary_rank(device_mesh),
        )

        if device_mesh is not None and device_mesh.mesh_dim_names is not None and "model" in device_mesh.mesh_dim_names:
            d_sae_local = sae.cfg.d_sae // device_mesh["model"].size()
        else:
            d_sae_local = sae.cfg.d_sae

        # Initialize tracking variables
        sample_result = {k: None for k in self.cfg.subsamples.keys()}
        if device_mesh is not None:
            act_times = torch.distributed.tensor.zeros(
                (sae.cfg.d_sae,),
                dtype=torch.long,
                device_mesh=device_mesh,
                placements=DimMap({"model": 0}).placements(device_mesh),
            )
            max_feature_acts = torch.distributed.tensor.zeros(
                (sae.cfg.d_sae,),
                dtype=sae.cfg.dtype,
                device_mesh=device_mesh,
                placements=DimMap({"model": 0}).placements(device_mesh),
            )
        else:
            act_times = torch.zeros((d_sae_local,), dtype=torch.long, device=sae.cfg.device)
            max_feature_acts = torch.zeros((d_sae_local,), dtype=sae.cfg.dtype, device=sae.cfg.device)
        mapper = KeyedDiscreteMapper()

        if isinstance(sae, CrossLayerTranscoder):
            sae.encode = partial(sae.encode_single_layer, layer=self.cfg.clt_layer)
            sae.prepare_input = partial(sae.prepare_input_single_layer, layer=self.cfg.clt_layer)
            sae.decoder_norm_per_feature = partial(sae.decoder_norm_per_feature, layer=self.cfg.clt_layer)
            sae.keep_only_decoders_for_layer_from(self.cfg.clt_layer)
            torch.cuda.empty_cache()

        # Process activation batches
        for batch in activation_stream:
            # Reshape meta to zip outer dimensions to inner
            meta = {k: [m[k] for m in batch["meta"]] for k in batch["meta"][0].keys()}

            # Get feature activations from SAE
            x, encoder_kwargs, _ = sae.prepare_input(batch)
            tokens = batch["tokens"]
            feature_acts: torch.Tensor = sae.encode(x, **encoder_kwargs)
            if isinstance(feature_acts, DTensor):
                assert device_mesh is not None, "Device mesh is required for DTensor feature activations"
                if device_mesh is not feature_acts.device_mesh:
                    feature_acts = DTensor.from_local(
                        feature_acts.redistribute(
                            placements=DimMap({"head": -1, "model": -1}).placements(feature_acts.device_mesh)
                        ).to_local(),
                        device_mesh,
                        placements=DimMap({"model": -1}).placements(device_mesh),
                    )
                    # TODO: Remove this once redistributing across device meshes is supported
                feature_acts = feature_acts.redistribute(placements=DimMap({"model": -1}).placements(device_mesh))
                if not isinstance(tokens, DTensor):
                    tokens = DTensor.from_local(tokens, device_mesh, placements=DimMap({}).placements(device_mesh))
            if isinstance(sae, CrossCoder):
                feature_acts = feature_acts.amax(dim=-2)
            if isinstance(sae, LowRankSparseAttention) and sae.cfg.skip_bos:
                feature_acts[:, 0, :] = 0
            assert feature_acts.shape == (tokens.shape[0], tokens.shape[1], sae.cfg.d_sae), (
                f"feature_acts.shape: {feature_acts.shape}, expected: {(tokens.shape[0], tokens.shape[1], sae.cfg.d_sae)}"
            )

            # Compute ignore token masks
            ignore_token_masks = self.compute_ignore_token_masks(tokens, self.cfg.ignore_token_ids)
            feature_acts *= rearrange(ignore_token_masks, "batch_size n_ctx -> batch_size n_ctx 1")

            # Update activation statistics
            active_feature_count = feature_acts.gt(0.0).sum(dim=[0, 1])
            act_times += active_feature_count
            max_feature_acts = torch.max(max_feature_acts, feature_acts.max(dim=0).values.max(dim=0).values)

            # Apply discrete mapper encoding only to string metadata, keep others as-is
            meta_device = feature_acts.to_local().device if isinstance(feature_acts, DTensor) else feature_acts.device
            discrete_meta = {}
            for k, v in meta.items():
                if all(isinstance(item, str) for item in v):
                    # Apply discrete mapper encoding to string metadata
                    discrete_meta[k] = torch.tensor(mapper.encode(k, v), device=meta_device, dtype=torch.int32)
                else:
                    # Keep non-string metadata as-is (assuming they are already tensors or can be converted)
                    discrete_meta[k] = torch.tensor(v, device=meta_device)
            if device_mesh is not None:
                discrete_meta = {
                    k: DTensor.from_local(
                        local_tensor=repeat(v, "batch_size -> batch_size d_sae", d_sae=d_sae_local),
                        device_mesh=device_mesh,
                        placements=DimMap({"model": 1}).placements(device_mesh),
                    )
                    for k, v in discrete_meta.items()
                }
            sample_result = self._process_batch(
                feature_acts, discrete_meta, sample_result, max_feature_acts, device_mesh
            )

            # Update progress
            n_tokens_current = tokens.numel()
            n_tokens += n_tokens_current
            n_analyzed_tokens += cast(int, ignore_token_masks.int().sum().item())
            pbar.update(n_tokens_current)
            if n_tokens >= self.cfg.total_analyzing_tokens:
                break

        pbar.close()

        # Filter out None values and format final per-feature results
        sample_result = {k: v for k, v in sample_result.items() if v is not None}
        sample_result = {
            name: {k: to_local(v) for k, v in subsample.items()} for name, subsample in sample_result.items()
        }

        return self.get_post_analysis_func(sae.cfg.sae_type).process(
            sae=sae,
            act_times=to_local(act_times),
            n_analyzed_tokens=n_analyzed_tokens,
            max_feature_acts=to_local(max_feature_acts),
            sample_result=sample_result,
            mapper=mapper,
            device_mesh=device_mesh,
            activation_factory=activation_factory,
            activation_factory_process_kwargs=activation_factory_process_kwargs,
        )
