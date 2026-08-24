import json
import os
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal, Optional, Tuple

import torch
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    WithJsonSchema,
)
from typing_extensions import override

from .utils.huggingface import parse_pretrained_name_or_path
from .utils.misc import (
    convert_str_to_torch_dtype,
    convert_torch_dtype_to_str,
)


class BaseConfig(BaseModel):
    pass


class BaseModelConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)  # allow parsing torch.dtype

    device: str = Field(default="cpu", exclude=True)

    dtype: Annotated[
        torch.dtype,
        BeforeValidator(lambda v: convert_str_to_torch_dtype(v) if isinstance(v, str) else v),
        PlainSerializer(convert_torch_dtype_to_str),
        WithJsonSchema(
            {
                "type": "string",
            },
            mode="serialization",
        ),
    ] = Field(default=torch.float32, exclude=True, validate_default=False)


class BaseSAEConfig(BaseModelConfig, ABC):
    """
    Base class for SAE configs.
    Initializer will initialize SAE based on config type.
    So this class should not be used directly but only as a base config class for other SAE variants like SAEConfig, CrossCoderConfig, etc.
    """

    sae_type: Literal["sae", "crosscoder", "clt", "lorsa", "molt"]
    d_model: int
    expansion_factor: int | float
    use_decoder_bias: bool = True
    act_fn: Literal["relu", "jumprelu", "topk", "batchtopk", "batchlayertopk", "layertopk", "mish"] = "relu"
    norm_activation: str = "dataset-wise"
    sparsity_include_decoder_norm: bool = True
    top_k: int = 50
    sae_pretrained_name_or_path: Optional[str] = None
    strict_loading: bool = True # TODO Set to true
    use_triton_kernel: bool = False
    sparsity_threshold_for_triton_spmm_kernel: float = 0.996
    sparsity_threshold_for_csr: float = 0.05
    circuit_tracing_mode: bool = False
    # anthropic jumprelu
    jumprelu_threshold_window: float = 2.0
    promote_act_fn_dtype: Annotated[
        torch.dtype | None,
        BeforeValidator(lambda v: convert_str_to_torch_dtype(v) if isinstance(v, str) else v),
        PlainSerializer(convert_torch_dtype_to_str),
        WithJsonSchema(
            {
                "type": ["string", "null"],
            },
            mode="serialization",
        ),
    ] = Field(default=None, exclude=True, validate_default=False)

    @property
    def d_sae(self) -> int:
        result = self.d_model * self.expansion_factor
        # Ensure result is integer (round if expansion_factor is float)
        return int(round(result))

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path: str, strict_loading: bool = True, **kwargs):
        """Load the SAEConfig from a pretrained SAE name or path. Config is read from <pretrained_name_or_path>/hyperparams.json.

        Args:
            sae_path (str): The path to the pretrained SAE.
            **kwargs: Additional keyword arguments to pass to the SAEConfig constructor.
        """
        path = parse_pretrained_name_or_path(pretrained_name_or_path)
        with open(os.path.join(path, "config.json"), "r") as f:
            sae_config = json.load(f)
        sae_config["sae_pretrained_name_or_path"] = pretrained_name_or_path
        sae_config["strict_loading"] = strict_loading
        return cls.model_validate({**sae_config, **kwargs})

    def save_hyperparameters(self, sae_path: Path | str, remove_loading_info: bool = True):
        assert os.path.exists(sae_path), f"{sae_path} does not exist. Unable to save hyperparameters."
        d = self.model_dump()
        if remove_loading_info:
            d.pop("sae_pretrained_name_or_path", None)
            d.pop("strict_loading", None)

        with open(os.path.join(sae_path, "config.json"), "w") as f:
            json.dump(d, f, indent=4)

    @property
    @abstractmethod
    def associated_hook_points(self) -> list[str]:
        pass


class SAEConfig(BaseSAEConfig):
    sae_type: Literal["sae", "crosscoder", "clt", "lorsa", "molt"] = "sae"
    hook_point_in: str
    hook_point_out: str
    use_glu_encoder: bool = False

    use_auxk: bool = False
    dead_threshold: int = 100_000  # Threshold for marking latents as dead (10 million tokens)
    k_aux: int = 512  # Number of dead latents to use for AuxK loss
    aux_coefficient: float = 1.0 / 32  # Weight coefficient for AuxK loss (alpha)


    @property
    def associated_hook_points(self) -> list[str]:
        return [self.hook_point_in, self.hook_point_out]


class LorsaConfig(BaseSAEConfig):
    """Configuration for Low Rank Sparse Attention."""

    sae_type: Literal["sae", "crosscoder", "clt", "lorsa", "molt"] = "lorsa"

    hook_point_in: str
    hook_point_out: str

    # Attention dimensions
    n_qk_heads: int
    d_qk_head: int
    positional_embedding_type: Literal["rotary", "none"] = "none"
    rotary_dim: int
    rotary_base: int = 10000
    rotary_adjacent_pairs: bool = True
    rotary_scale: int = 1
    use_NTK_by_parts_rope: bool = False
    NTK_by_parts_factor: float = 1.0
    NTK_by_parts_low_freq_factor: float = 1.0
    NTK_by_parts_high_freq_factor: float = 1.0
    old_context_len: int = 64

    n_ctx: int
    skip_bos: bool = False

    # Attention settings
    attn_scale: Optional[float] = None
    use_post_qk_ln: bool = False
    normalization_type: Literal["LN", "RMS"] | None = None
    eps: float = 1e-6
    
    # SmolGen settings
    attn_scale: Optional[float] = None
    use_smolgen: bool = False
    # Learnable attention scale
    use_learnable_attn_scale: bool = False

    # Auxk Loss
    use_auxk: bool = False
    dead_threshold: int = 100_000
    k_aux: int = 512  # Number of dead latents to use for AuxK loss
    aux_coefficient: float = 1.0 / 32  # Weight coefficient for AuxK loss (alpha)

    @property
    def n_ov_heads(self) -> int:
        return self.d_sae

    @property
    def ov_group_size(self) -> int:
        return self.n_ov_heads // self.n_qk_heads

    @property
    def associated_hook_points(self) -> list[str]:
        """All hook points used by Lorsa."""
        return [self.hook_point_in, self.hook_point_out]

    def model_post_init(self, __context):
        super().model_post_init(__context)
        assert self.hook_point_in is not None and self.hook_point_out is not None, (
            "hook_point_in and hook_point_out must be set"
        )
        assert self.hook_point_in != self.hook_point_out, "hook_point_in and hook_point_out must be different"
        assert self.n_ov_heads % self.n_qk_heads == 0, "n_ov_heads must be divisible by n_qk_heads"

        if self.attn_scale is None:
            self.attn_scale = self.d_qk_head**0.5


class CLTConfig(BaseSAEConfig):
    """Configuration for Cross Layer Transcoder (CLT).

    A CLT consists of L encoders and L(L+1)/2 decoders where each encoder at layer L
    reads from the residual stream at that layer and can decode to layers L through L-1.
    """

    sae_type: Literal["sae", "crosscoder", "clt", "lorsa", "molt"] = "clt"
    act_fn: Literal["relu", "jumprelu", "topk", "batchtopk", "batchlayertopk", "layertopk", "mish"] = "relu"
    init_cross_layer_decoder_all_zero: bool = False
    hook_points_in: list[str]
    """List of hook points to capture input activations from, one for each layer."""
    hook_points_out: list[str]
    """List of hook points to capture output activations from, one for each layer."""
    decode_with_csr: bool = False
    """Whether to decode with CSR matrices. If `True`, will use CSR matrices for decoding. If `False`, will use dense matrices for decoding."""

    @property
    def n_layers(self) -> int:
        """Number of layers in the CLT."""
        return len(self.hook_points_in)

    @property
    def n_decoders(self) -> int:
        """Number of decoders in the CLT."""
        return self.n_layers * (self.n_layers + 1) // 2

    @property
    def associated_hook_points(self) -> list[str]:
        """All hook points used by the CLT."""
        return self.hook_points_in + self.hook_points_out

    def model_post_init(self, __context):
        super().model_post_init(__context)
        assert len(self.hook_points_in) == len(self.hook_points_out), (
            "Number of input and output hook points must match"
        )


class MOLTConfig(BaseSAEConfig):
    """Configuration for Mixture of Linear Transforms (MOLT).

    MOLT is a more efficient alternative to transcoders that sparsely replaces
    MLP computation in transformers. It converts dense MLP layers into sparse,
    interpretable linear transforms.
    """

    sae_type: Literal["sae", "crosscoder", "clt", "lorsa", "molt"] = "molt"
    hook_point_in: str
    """Hook point to capture input activations from."""
    hook_point_out: str
    """Hook point to output activations to."""
    rank_distribution: dict[int, int] = Field(default_factory=lambda: {4: 1, 8: 2, 16: 4, 32: 8, 64: 16})
    """Dictionary mapping rank values to their integer ratios. 
    Keys are rank values, values are integer ratios that will be automatically normalized to proportions.
    Example: {4: 1, 8: 2, 16: 4, 32: 8, 64: 16} means ratio 1:2:4:8:16 which means a proportion of 1/32, 2/32, 4/32, 8/32, 16/32, 
    which will be normalized to proportions automatically."""
    model_parallel_size_training: int = 1
    """Number of model parallel devices for distributed training. Distinct from model_parallel_size_running which is the number of model parallel devices in both training and inference."""

    def model_post_init(self, __context):
        super().model_post_init(__context)
        # Validate ratios
        assert self.rank_distribution, "rank_distribution cannot be empty"

        total_ratio = sum(self.rank_distribution.values())
        assert total_ratio > 0, f"Total ratio must be positive, got {total_ratio}"

        for rank, ratio in self.rank_distribution.items():
            assert ratio > 0, f"Ratio for rank {rank} must be positive, got {ratio}"

        # Store normalized proportions for internal use
        self._normalized_proportions = {rank: ratio / total_ratio for rank, ratio in self.rank_distribution.items()}

    def generate_rank_assignments(self) -> list[int]:
        """Generate rank assignment for each of the d_sae linear transforms.

        Returns:
            List of rank assignments for each transform.
            For example: [1, 1, 1, 1, 2, 2, 4].
            For distributed case, this method ensures that each rank type is divisible by model_parallel_size_training.
        """
        # Validate rank distribution
        assert self.rank_distribution, "rank_distribution cannot be empty"

        # Calculate base d_sae
        base_d_sae = int(round(self.d_model * self.expansion_factor))

        # For distributed training, use special logic to ensure consistency
        if self.model_parallel_size_training > 1:
            return self._generate_distributed_rank_assignments(base_d_sae)
        else:
            return self._generate_rank_assignments_single_gpu(base_d_sae)

    def _generate_rank_assignments_single_gpu(self, base_d_sae: int) -> list[int]:
        """Generate rank assignments for single GPU training.

        Args:
            base_d_sae: Target number of total transforms

        Returns:
            List of rank assignments for each transform.
        """
        assignments = []

        # Distribute transforms based on normalized proportions
        for rank, proportion in sorted(self._normalized_proportions.items()):
            count = int(base_d_sae * proportion)
            assignments.extend([rank] * count)

        # Handle any remaining transforms due to rounding
        while len(assignments) < base_d_sae:
            # Assign remaining to the most common rank (by original ratio)
            most_common_rank = max(self.rank_distribution.keys(), key=lambda k: self.rank_distribution[k])
            assignments.append(most_common_rank)

        # Truncate if we have too many (shouldn't happen with proper proportions)
        assignments = assignments[:base_d_sae]

        # Verify we have exactly base_d_sae assignments
        assert len(assignments) == base_d_sae, f"Expected {base_d_sae} assignments, got {len(assignments)}"

        return assignments

    def _generate_distributed_rank_assignments(self, base_d_sae: int) -> list[int]:
        """Generate rank assignments optimized for distributed training.

        Ensures each rank type has count divisible by model_parallel_size_training.

        Args:
            base_d_sae: Target number of total transforms

        Returns:
            List of rank assignments for each transform.
        """
        assignments = []
        total_ratio = sum(self.rank_distribution.values())

        # Ensure minimum requirement: each rank gets at least model_parallel_size_training
        # transforms
        min_total_needed = len(self.rank_distribution) * self.model_parallel_size_training
        assert base_d_sae >= min_total_needed, (
            f"base_d_sae ({base_d_sae}) must be >= min_total_needed "
            f"({min_total_needed}) for distributed training with "
            f"{len(self.rank_distribution)} rank types"
        )

        # Calculate proportional distribution
        for rank in sorted(self.rank_distribution.keys()):
            rank_ratio = self.rank_distribution[rank]
            raw_count = int(base_d_sae * rank_ratio / total_ratio)

            # Ensure count is divisible by model_parallel_size_training
            count = max(
                self.model_parallel_size_training,  # minimum requirement
                (raw_count // self.model_parallel_size_training) * self.model_parallel_size_training,
            )
            assignments.extend([rank] * count)

        # Handle any remaining transforms due to rounding
        while len(assignments) < base_d_sae:
            # Assign remaining to the most common rank (by original ratio)
            most_common_rank = max(self.rank_distribution.keys(), key=lambda k: self.rank_distribution[k])
            # Add model_parallel_size_training transforms at a time for divisibility
            remaining = base_d_sae - len(assignments)
            to_add = min(self.model_parallel_size_training, remaining)
            assignments.extend([most_common_rank] * to_add)

        # Truncate if we have too many (shouldn't happen normally)
        assignments = assignments[:base_d_sae]

        # Verify divisibility constraint
        rank_counts = {}
        for rank in assignments:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1

        for rank, count in rank_counts.items():
            assert count % self.model_parallel_size_training == 0, (
                f"Rank {rank} count {count} not divisible by "
                f"model_parallel_size_training {self.model_parallel_size_training}"
            )

        return assignments

    def get_local_rank_assignments(self, local_rank: int, model_parallel_size_running: int) -> list[int]:
        """Get rank assignments for a specific local device in distributed running (both training and inference).

        Each device gets all rank groups, with each group evenly divided across devices.
        This ensures consistent encoder/decoder sharding without feature_acts redistribution.

        Args:
            local_rank: The local rank of this process
            model_parallel_size_running: Number of model parallel devices in running (training and inference)

        Returns:
            List of rank assignments for this local device
            For example:
            global_rank_assignments = [1, 1, 2, 2], model_parallel_size_running = 2 -> local_rank_assignments = [1, 2]
        """
        global_rank_counts = {rank: self.generate_rank_assignments().count(rank) for rank in self.available_ranks}

        # Each device gets count/model_parallel_size_running transforms of each rank type
        local_assignments = []
        for rank in sorted(self.rank_distribution.keys()):
            global_count = global_rank_counts[rank]

            # Verify even division (should be guaranteed by _generate_distributed_rank_assignments)
            assert global_count % model_parallel_size_running == 0, (
                f"Rank {rank} global count {global_count} not divisible by "
                f"model_parallel_size_running {model_parallel_size_running}"
            )

            local_count = global_count // model_parallel_size_running

            # Add local_count transforms of this rank type
            local_assignments.extend([rank] * local_count)

        return local_assignments

    @property
    @override
    def d_sae(self) -> int:
        """Calculate d_sae based on rank assignments with padding for distributed training."""
        # Generate rank assignments and return the length
        rank_assignments = self.generate_rank_assignments()
        return len(rank_assignments)

    @property
    def available_ranks(self) -> list[int]:
        """Get sorted list of available ranks."""
        return sorted(self.rank_distribution.keys())

    @property
    def num_rank_types(self) -> int:
        """Number of different rank types."""
        return len(self.rank_distribution)

    @property
    def associated_hook_points(self) -> list[str]:
        return [self.hook_point_in, self.hook_point_out]


class CrossCoderConfig(BaseSAEConfig):
    sae_type: Literal["sae", "crosscoder", "clt", "lorsa", "molt"] = "crosscoder"
    hook_points: list[str]

    @property
    def associated_hook_points(self) -> list[str]:
        return self.hook_points

    @property
    def n_heads(self) -> int:
        return len(self.hook_points)


class InitializerConfig(BaseConfig):
    bias_init_method: Literal["all_zero", "geometric_median"] = "all_zero"
    decoder_uniform_bound: float = 1.0
    encoder_uniform_bound: float = 1.0
    init_encoder_with_decoder_transpose: bool = True
    init_encoder_with_decoder_transpose_factor: float = 1.0
    init_log_jumprelu_threshold_value: float | None = None
    grid_search_init_norm: bool = False
    initialize_W_D_with_active_subspace: bool = False
    d_active_subspace: int | None = None
    initialize_lorsa_with_mhsa: bool | None = None
    model_layer: int | None = None
    init_encoder_bias_with_mean_hidden_pre: bool = False
    initialize_lorsa_smolgen_from_encoder: bool = False
    initialize_W_D_with_mhsa: bool = False
    initialize_lorsa_attn_scale_from_encoder: bool = False


class TrainerConfig(BaseConfig):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    l1_coefficient: float | None = 0.00008
    l1_coefficient_warmup_steps: int | float = 0.1
    lp_coefficient: float | None = None
    amp_dtype: Annotated[
        torch.dtype | None,
        BeforeValidator(lambda v: convert_str_to_torch_dtype(v) if isinstance(v, str) else v),
        PlainSerializer(convert_torch_dtype_to_str),
        WithJsonSchema(
            {
                "type": ["string", "null"],
            },
            mode="serialization",
        ),
    ] = Field(default=torch.bfloat16, exclude=True, validate_default=False)
    sparsity_loss_type: Literal["power", "tanh", "tanh-quad", None] = None
    tanh_stretch_coefficient: float = 4.0
    frequency_scale: float = 0.01
    p: int = 1
    initial_k: int | float | None = None
    k_warmup_steps: int | float = 0.1
    k_cold_booting_steps: int | float = 0
    k_schedule_type: Literal["linear", "exponential"] = "linear"
    k_exponential_factor: float = 3.0
    use_batch_norm_mse: bool = True
    skip_metrics_calculation: bool = False
    gradient_accumulation_steps: int = 1

    lr: float | dict[str, float] = 0.0004
    betas: Tuple[float, float] = (0.9, 0.999)
    optimizer_class: Literal["adam", "sparseadam"] = "adam"
    optimizer_foreach: bool = True
    lr_scheduler_name: Literal[
        "constant",
        "constantwithwarmup",
        "linearwarmupdecay",
        "cosineannealing",
        "cosineannealingwarmup",
        "exponentialwarmup",
    ] = "constantwithwarmup"
    lr_end_ratio: float = 1 / 32
    lr_warm_up_steps: int | float = 5000
    lr_cool_down_steps: int | float = 0.2
    jumprelu_lr_factor: float = 1.0
    clip_grad_norm: float = 0.0
    feature_sampling_window: int = 1000
    total_training_tokens: int = 300_000_000

    log_frequency: int = 1000
    eval_frequency: int = 1000
    n_checkpoints: int = 10
    check_point_save_mode: Literal["log", "linear"] = "log"

    from_pretrained_path: Optional[str] = None
    exp_result_path: Path = Path("results")

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.exp_result_path.mkdir(parents=True, exist_ok=True)
        self.exp_result_path.joinpath("checkpoints").mkdir(parents=True, exist_ok=True)
        assert self.lr_end_ratio <= 1, "lr_end_ratio must be in 0 to 1 (inclusive)."

        if self.from_pretrained_path is not None:
            assert os.path.exists(self.from_pretrained_path), (
                f"from_pretrained_path {self.from_pretrained_path} does not exist"
            )


class EvalConfig(BaseConfig):
    feature_sampling_window: int = 1000
    total_eval_tokens: int = 1000000
    use_cached_activations: bool = False
    device: str = "cpu"
    fold_activation_scale: bool = True
    """Whether to fold the activation scale into the SAE model"""


class GraphEvalConfig(BaseConfig):
    max_n_logits: int = 2
    # How many logits to attribute from, max. We attribute to min(max_n_logits, n_logits_to_reach_desired_log_prob); see below for the latter

    desired_logit_prob: float = 0.95
    # Attribution will attribute from the minimum number of logits needed to reach this probability mass (or max_n_logits, whichever is lower)

    max_feature_nodes: int = 4096
    # Only attribute from this number of feature nodes, max. Lower is faster, but you will lose more of the graph. None means no limit.

    batch_size: int = 64
    # Batch size when attributing

    offload: Literal[None, "disk", "cpu"] = None
    # Offload various parts of the model during attribution to save memory. Can be 'disk', 'cpu', or None (keep on GPU)

    start_from: int = 0


class DatasetConfig(BaseConfig):
    dataset_name_or_path: str = "openwebtext"
    cache_dir: Optional[str] = None
    is_dataset_on_disk: bool = False


class ActivationFactorySource(BaseModel):
    type: str
    """ The type of the source. Used to determine the source of activations in deserialization. """
    name: str
    sample_weights: float = 1.0
    """ The sample weights to use for the source. Will be used to randomly pull tokens from the source when multiple sources are present. """


class ActivationFactoryDatasetSource(ActivationFactorySource):
    type: str = "dataset"
    is_dataset_tokenized: bool = False
    """ Whether the dataset is tokenized. Non-tokenized datasets should have records with fields `text`, `images`, etc. Tokenized datasets should have records with fields `tokens`, which could contain either padded or non-padded tokens. """
    prepend_bos: bool = True
    """ Whether to prepend the BOS token to each record when tokenizing. """


class ActivationFactoryActivationsSource(ActivationFactorySource):
    model_config = ConfigDict(arbitrary_types_allowed=True)  # allow parsing torch.dtype

    type: str = "activations"
    path: str | dict[str, Path]
    """ The path to the cached activations. """
    device: str = "cpu"
    """ The device to load the activations on. """
    dtype: Optional[
        Annotated[
            torch.dtype,
            BeforeValidator(lambda v: convert_str_to_torch_dtype(v) if isinstance(v, str) else v),
            PlainSerializer(convert_torch_dtype_to_str),
            WithJsonSchema(
                {
                    "type": "string",
                },
                mode="serialization",
            ),
        ]
    ] = None
    """ We might want to convert presaved bf16 activations to fp32"""
    num_workers: int = 4
    """ The number of workers to use for loading the activations. """
    prefetch: Optional[int] = 8
    """ The number of chunks to prefetch."""


class ActivationFactoryTarget(Enum):
    TOKENS = "tokens"
    """ Output non-padded and non-truncated tokens. """
    ACTIVATIONS_2D = "activations-2d"
    """ Output activations in `(batch_size, seq_len, d_model)` shape. Tokens are padded and truncated to the same length. """
    ACTIVATIONS_1D = "activations-1d"
    """ Output activations in `(n_filtered_tokens, d_model)` shape. Tokens are filtered in this stage. """

    @property
    def stage(self) -> int:
        return {
            ActivationFactoryTarget.TOKENS: 0,
            ActivationFactoryTarget.ACTIVATIONS_2D: 1,
            ActivationFactoryTarget.ACTIVATIONS_1D: 2,
        }[self]

    def __lt__(self, other: "ActivationFactoryTarget") -> bool:
        return self.stage < other.stage

    def __le__(self, other: "ActivationFactoryTarget") -> bool:
        return self.stage <= other.stage


class BufferShuffleConfig(BaseConfig):
    perm_seed: int = 42
    """ Perm seed for aligned permutation for generating activations. If `None`, will not use manual seed for Generator. """
    generator_device: Optional[str] = None
    """ The device to be assigned for the torch.Generator. If 'None', generator will be initialized on cpu as pytorch default. """


class ActivationFactoryConfig(BaseConfig):
    model_config = ConfigDict(arbitrary_types_allowed=True)  # allow parsing torch.dtype

    sources: list[ActivationFactoryDatasetSource | ActivationFactoryActivationsSource]
    """ List of sources to use for activations. Can be a dataset or a path to activations. """
    target: ActivationFactoryTarget
    """ The target to produce. """
    hook_points: list[str]
    """ The hook points to capture activations from. """
    num_workers: int = 4
    """ The number of workers to use for loading the dataset. """
    context_size: Optional[int] = None
    """ The context size to use for generating activations. All tokens will be padded or truncated to this size. If `None`, will not pad or truncate tokens. This may lead to some error when re-batching activations of different context sizes."""
    model_batch_size: int = 1
    """ The batch size to use for generating activations. """
    batch_size: Optional[int] = Field(
        default_factory=lambda validated_model: 64
        if validated_model["target"] == ActivationFactoryTarget.ACTIVATIONS_1D
        else None
    )
    """ The batch size to use for outputting `activations-1d`. """
    override_dtype: Optional[
        Annotated[
            torch.dtype,
            BeforeValidator(lambda v: convert_str_to_torch_dtype(v) if isinstance(v, str) else v),
            PlainSerializer(convert_torch_dtype_to_str),
            WithJsonSchema(
                {
                    "type": "string",
                },
                mode="serialization",
            ),
        ]
    ] = None
    """ The dtype to use for outputting activations. If `None`, will not override the dtype. """
    buffer_size: Optional[int] = Field(
        default_factory=lambda validated_model: 500_000
        if validated_model["target"] == ActivationFactoryTarget.ACTIVATIONS_1D
        else None
    )
    """ Buffer size for online shuffling. If `None`, no shuffling will be performed. """
    buffer_shuffle: Optional[BufferShuffleConfig] = None
    """" Manual seed and device of generator for generating randomperm in buffer. """
    ignore_token_ids: Optional[list[int]] = None
    """ Tokens to ignore in the activations. """


class LanguageModelConfig(BaseModelConfig):
    model_name: str = "gpt2"
    """ The name of the model to use. """
    model_from_pretrained_path: Optional[str] = None
    """ The path to the pretrained model. If `None`, will use the model from HuggingFace. """
    use_flash_attn: bool = False
    """ Whether to use Flash Attention. """
    cache_dir: Optional[str] = None
    """ The directory of the HuggingFace cache. Should have the same effect as `HF_HOME`. """
    d_model: int = 768
    """ The dimension of the model. """
    local_files_only: bool = False
    """ Whether to only load the model from the local files. Should have the same effect as `HF_HUB_OFFLINE=1`. """
    max_length: int = 2048
    """ The maximum length of the input. """
    backend: Literal["huggingface", "transformer_lens", "auto"] = "auto"
    """ The backend to use for the language model. """
    load_ckpt: bool = True
    tokenizer_only: bool = False
    """ Whether to only load the tokenizer. """
    prepend_bos: bool = True
    """ Whether to prepend the BOS token to the input. """
    bos_token_id: Optional[int] = None
    """ The ID of the BOS token. If `None`, will use the default BOS token. """
    eos_token_id: Optional[int] = None
    """ The ID of the EOS token. If `None`, will use the default EOS token. """
    pad_token_id: Optional[int] = None
    """ The ID of the padding token. If `None`, will use the default padding token. """

    @staticmethod
    def from_pretrained_sae(pretrained_name_or_path: str, **kwargs):
        """Load the LanguageModelConfig from a pretrained SAE name or path. Config is read from <pretrained_name_or_path>/lm_config.json.

        Args:
            sae_path (str): The path to the pretrained SAE.
            **kwargs: Additional keyword arguments to pass to the LanguageModelConfig constructor.
        """
        path = parse_pretrained_name_or_path(pretrained_name_or_path)
        with open(os.path.join(path, "lm_config.json"), "r") as f:
            lm_config = json.load(f)
        return LanguageModelConfig.model_validate(lm_config, **kwargs)

    def save_lm_config(self, sae_path: str):
        assert os.path.exists(sae_path), f"{sae_path} does not exist. Unable to save LanguageModelConfig."

        d = self.model_dump()
        with open(os.path.join(sae_path, "lm_config.json"), "w") as f:
            json.dump(d, f, indent=4)


class LLaDAConfig(LanguageModelConfig):
    mask_ratio: float = 0.0
    mdm_mask_token_id: int = 126336
    prepend_bos: bool = False
    calculate_logits: bool = False


class ActivationWriterConfig(BaseConfig):
    hook_points: list[str]
    """ The hook points to capture activations from. """
    total_generating_tokens: Optional[int] = None
    """ The total number of tokens to generate. If `None`, will write all activations to disk. """
    n_samples_per_chunk: Optional[int] = None
    """ The number of samples to write to disk per chunk. If `None`, will not further batch the activations. """
    cache_dir: str | Path = Path("activations")
    """ The directory to save the activations. """
    format: Literal["pt", "safetensors"] = "safetensors"
    num_workers: Optional[int] = None
    """ The number of workers to use for writing the activations. If `None`, will not use multi-threaded writing. """


class FeatureAnalyzerConfig(BaseConfig):
    model_config = ConfigDict(arbitrary_types_allowed=True)  # allow parsing torch.dtype
    total_analyzing_tokens: int
    """ Total number of tokens to analyze """

    ignore_token_ids: Optional[list[int]] = None
    """ Tokens to ignore in the activations. """

    subsamples: dict[str, dict[str, int | float]] = Field(
        default_factory=lambda: {
            "top_activations": {"proportion": 1.0, "n_samples": 10},
            "non_activating": {
                "proportion": 0.3,
                "n_samples": 20,
                "max_length": 50,
            },
        }
    )
    """ Dictionary mapping subsample names to their parameters:
        - `proportion`: Proportion of max activation to consider
        - `n_samples`: Number of samples to keep
        - `max_length`: Maximum length of the sample
    """

    clt_layer: int | None = None
    """ Layer to analyze for CLT. Provided iff analyzing CLT. """


class DirectLogitAttributorConfig(BaseConfig):
    top_k: int = 10
    """ The number of top tokens to attribute to. """

    clt_layer: int | None = None
    """ Layer to analyze for CLT. Provided iff analyzing CLT. """


class WandbConfig(BaseConfig):
    wandb_project: str = "gpt2-sae-training"
    exp_name: Optional[str] = None
    wandb_entity: Optional[str] = None


class MongoDBConfig(BaseConfig):
    mongo_uri: str = Field(default_factory=lambda: os.environ.get("MONGO_URI", "mongodb://localhost:27017/"))
    mongo_db: str = Field(default_factory=lambda: os.environ.get("MONGO_DB", "mechinterp"))
