import json
import math
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from importlib.metadata import version
from pathlib import Path
from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    Self,
    Union,
    cast,
    overload,
)

import einops
import safetensors.torch as safe
import torch
import torch.distributed.checkpoint as dcp
import torch.distributed.tensor
from huggingface_hub import create_repo, hf_hub_download, snapshot_download, upload_folder
from jaxtyping import Float
from safetensors import safe_open
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.experimental import local_map
from transformer_lens.hook_points import HookedRootModule

from llamascopium.activation_functions import JumpReLU
from llamascopium.backend.language_model import LanguageModelConfig
from llamascopium.config import BaseModelConfig
from llamascopium.models.protocols import DatasetNormStandardizable, NormComputing
from llamascopium.utils.auto import PretrainedSAEType, auto_infer_pretrained_sae_type
from llamascopium.utils.distributed import DimMap, distributed_topk, is_primary_rank, item, mesh_dim_size
from llamascopium.utils.distributed.utils import execute_and_broadcast
from llamascopium.utils.logging import get_distributed_logger
from llamascopium.utils.math import topk
from llamascopium.utils.tensor_specs import TensorSpecs, apply_token_mask
from llamascopium.utils.timer import timer

logger = get_distributed_logger("sparse_dictionary")


SAE_TYPE_TO_CONFIG_CLASS: dict[str, Any] = {}
SAE_TYPE_TO_MODEL_CLASS: dict[str, Any] = {}


def register_sae_config(name: str):
    """Register a sparse dictionary config class, to make it possible to automatically determine the correct config class based on the type. The registered config class should inherit from `SparseDictionaryConfig` and have a `sae_type` field that is the same as the name. This function should be used together with `register_sae_model`.

    Example usage:
    ```python
    @register_sae_config("custom_sae")
    class CustomSAEConfig(SparseDictionaryConfig):
        sae_type: str = "custom_sae"
        ...
    ```
    """

    def _register(cls):
        SAE_TYPE_TO_CONFIG_CLASS[name] = cls
        return cls

    return _register


def register_sae_model(name):
    """Register a sparse dictionary model class, to make it possible to automatically determine the correct model class based on the type. The registered model class should inherit from `SparseDictionary`. This function should be used together with `register_sae_config`.

    Example usage:
    ```python
    @register_sae_model("custom_sae")
    class CustomSAE(SparseDictionary):
        ...
    ```
    """

    def _register(cls):
        SAE_TYPE_TO_MODEL_CLASS[name] = cls
        return cls

    return _register


class SparseDictionaryConfig(BaseModelConfig, ABC):
    """
    Base class for sparse dictionary configs with common settings that are able to apply to various sparse dictionary variants. This class should not be used directly but only as a base config class for other sparse dictionary variants like SAEConfig, CrosscoderConfig, etc.
    """

    sae_type: str
    """The type of the sparse dictionary. Must be one of the registered sparse dictionary types."""

    d_model: int
    """The dimension of the input/label activation space. In common settings where activations come from a transformer, this is the dimension of the model (may also known as hidden_size)."""

    expansion_factor: float
    """The expansion factor of the sparse dictionary. The hidden dimension of the sparse dictionary `d_sae` is `d_model * expansion_factor`."""

    use_decoder_bias: bool = True
    """Whether to use a bias term in the decoder. Including bias term may make it easier to train a better sparse dictionary, in exchange for increased architectural complexity."""

    act_fn: Literal["relu", "jumprelu", "topk", "batchtopk", "batchlayertopk", "layertopk"] = "relu"
    """The activation function to use for the sparse dictionary. Currently supported activation functions are `relu`, `jumprelu`, `topk`, `batchtopk`, `batchlayertopk`, and `layertopk`.
    
    - `relu`: ReLU activation function. Used in the most vanilla SAE settings.
    - `jumprelu`: JumpReLU activation function, adding a trainable element-wise threshold that pre-activations must pass to be activated, which is formally defined as $f(x) = \\max(0, x - \\theta)$ where $\\theta$ is the threshold. Proposed in [Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders](https://arxiv.org/abs/2407.14435).
    - `topk`: TopK activation function. Retains the top K activations per sample, zeroing out the rest. Proposed in [Scaling and evaluating sparse autoencoders](https://openreview.net/forum?id=tcsZt9ZNKD).
    - `batchtopk`: BatchTopK activation function. Batch TopK relaxes TopK function to batch-level, ing the top `k * batch_size` activations per batch and zeroing out the rest. This allows more adaptive allocation of latents on each sample. Proposed in [BatchTopK Sparse Autoencoders](https://arxiv.org/abs/2412.06410).
    - `batchlayertopk`: (For CrossLayerTranscoder only) Extension of BatchTopK to layer-and-batch-aware, retaining the top `k * batch_size * n_layers` activations per batch and layer and zeroing out the rest.
    - `layertopk`: (For CrossLayerTranscoder only) Extension of BatchTopK to layer-aware, retaining the top `k * n_layers` activations per layer and zeroing out the rest. Note that this activation function does not take batch dimension into account.
    """

    norm_activation: Literal["token-wise", "batch-wise", "dataset-wise", "inference"] = "dataset-wise"
    """The activation normalization strategy to use for the input/label activations. During call of [`normalize_activations`][llamascopium.models.sparse_dictionary.SparseDictionary.normalize_activations] (which will be called by the Trainer during training), the input/label activations will be normalized to an average norm of $\\sqrt{d_\\text{model}}$. This allows easier hyperparameter (mostly learning rate) transfer between different scale of model activations, since the MSE loss without normalization is proportional to the square of the activation norm.
    
    Different activation normalization strategy determines in what view the norm is *averaged*, with the following options:
    
    - `token-wise`: Norm is directly computed for activation from each token. No averaging is performed.
    - `batch-wise`: Norm is computed for each batch, then averaged over the batch dimension.
    - `dataset-wise`: Norm is computed from several samples from the activation. Compared to `batch-wise`, `dataset-wise` gives a fixed value of average norm for all activations, preserving the linearity of pre-activation encoding and decoding.
    - `inference`: No normalization is performed. A inference mode is produced after calling [`standardize_parameters_of_dataset_norm`][llamascopium.models.sparse_dictionary.SparseDictionary.standardize_parameters_of_dataset_norm] method, which folds the dataset-wise average norm into the weights and biases of the model. Switching to `inference` mode doesn't affect the encoding and decoding as a whole, that is, the reconstructed activations keep the same as the denormalized reconstructed activations in `dataset-wise` mode. However, the feature activations will reflect the activation scale. This allows real magnitude of feature activations to present during inference.
    """

    sparsity_include_decoder_norm: bool = True
    """Whether to include the decoder norm term in feature activation gating. If true, the pre-activation hidden states will be scaled by the decoder norm before applying the activation function, and then scale back after the activation function. Formally, considering activation function $f(x)$, an activation gating function $g(x)$ is defined as $g(x) = f(x) / x$ (element-wise division). When `sparsity_include_decoder_norm` is True, we replace $f(x)$ with $x \\cdot g(x \\cdot \\| W_\\text{dec} \\|)$. This effectively suppresses the training dynamics that model tries to increase the decoder norm in exchange of a smaller feature activation magnitude, resulting in lower sparsity loss (L1 norm)."""

    top_k: int = 50
    """The k value to use for the topk family of activation functions. For vanilla TopK, the L0 norm of the feature activations will be exactly equal to `top_k`."""

    sae_pretrained_name_or_path: str | None = None
    """Optional pretrained SAE path or identifier used to restore model weights."""

    strict_loading: bool = True
    """Whether to strictly enforce an exact state_dict key match when loading pretrained weights."""

    use_triton_kernel: bool = False
    """Whether to use the Triton SpMM kernel for the sparse matrix multiplication. Currently only supported for vanilla SAE."""

    sparsity_threshold_for_triton_spmm_kernel: float = 0.996
    """The sparsity threshold for the Triton SpMM kernel. Only when feature activation sparsity reaches this threshold, the Triton SpMM kernel will be used for the sparse matrix multiplication. This is useful for JumpReLU or TopK with a k annealing schedule, where the sparsity is not guaranteed throughout the training."""

    jumprelu_threshold_window: float = 2.0
    """The window size for the JumpReLU threshold. When pre-activations are element-wise in the window-neighborhood of the threshold, the threshold will begin to receive gradient. See [Anthropic's Circuits Update - January 2025](https://transformer-circuits.pub/2025/january-update/index.html#DL) for more details on how JumpReLU is optimized (where they refer to this window as $\\epsilon$)."""

    @property
    def d_sae(self) -> int:
        """The hidden dimension of the sparse dictionary. Calculated as `d_model * expansion_factor`."""
        assert math.isclose(
            self.d_model * self.expansion_factor, round(self.d_model * self.expansion_factor), rel_tol=0.0, abs_tol=1e-6
        ), "d_model * expansion_factor must be an integer"
        d_sae = int(self.d_model * self.expansion_factor)
        return d_sae

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path: str, strict_loading: bool = True, **kwargs):
        """Load the config of the sparse dictionary from a pretrained name or path. Config is read from <pretrained_name_or_path>/config.json (for local storage) or <repo_id>/<name>/config.json (for HuggingFace Hub).

        Args:
            pretrained_name_or_path (str): The path to the pretrained sparse dictionary.
            **kwargs (Any): Additional keyword arguments to pass to the config constructor.
        """
        sae_type = auto_infer_pretrained_sae_type(pretrained_name_or_path)
        if sae_type == PretrainedSAEType.LOCAL:
            path = os.path.join(pretrained_name_or_path, "config.json")
        elif sae_type == PretrainedSAEType.HUGGINGFACE:
            repo_id, name = pretrained_name_or_path.split(":")
            path = os.path.join(hf_hub_download(repo_id=repo_id, filename=f"{name}/config.json"), "config.json")
        elif sae_type == PretrainedSAEType.SAELENS:
            raise ValueError(
                "Currently not support directly generating config from SAELens. Try converting the whole model from SAELens through `from_saelens` or `from_pretrained` method instead."
            )
        else:
            raise ValueError(f"Unsupported pretrained type: {sae_type}")

        with open(path, "r") as f:
            sae_config = json.load(f)

        sae_config["sae_pretrained_name_or_path"] = pretrained_name_or_path
        sae_config["strict_loading"] = strict_loading

        if cls is SparseDictionaryConfig:
            cls = SAE_TYPE_TO_CONFIG_CLASS[sae_config["sae_type"]]

        return cls.model_validate({**sae_config, **kwargs})

    def save_hyperparameters(self, sae_path: str | Path, remove_loading_info: bool = True):
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
        """List of hook points used by the sparse dictionary, including all input and label hook points. This is used to retrieve useful data from the input activation source."""
        raise NotImplementedError("Subclasses must implement this method")

    @property
    @abstractmethod
    def hooks_in(self) -> list[str]:
        """List of hook points used by the sparse dictionary, including all input hook points. This is used to retrieve useful data from the input activation source."""
        raise NotImplementedError("Subclasses must implement this method")

    @property
    @abstractmethod
    def hooks_out(self) -> list[str]:
        """List of hook points used by the sparse dictionary, including all output hook points. This is used to retrieve useful data from the output activation source."""
        raise NotImplementedError("Subclasses must implement this method")


class SparseDictionary(HookedRootModule, ABC):
    """Abstract base class for all sparse dictionary models.

    This class defines the public interface for all sparse dictionary implementations.
    Concrete implementations should inherit from this class and selectively implement the
    following optional mixins based on the capabilities they support:

    - [`NormComputing`][llamascopium.models.protocols.NormComputing]: Weight norm computation (`encoder_norm`, `decoder_norm`, `decoder_bias_norm`)
    - [`NormConstrainable`][llamascopium.models.protocols.NormConstrainable]: Weight norm constraints (`set_decoder_to_fixed_norm`, `set_encoder_to_fixed_norm`, `transform_to_unit_decoder_norm`)
    - [`DatasetNormStandardizable`][llamascopium.models.protocols.DatasetNormStandardizable]: Dataset norm standardization (`standardize_parameters_of_dataset_norm`)
    - [`EncoderInitializable`][llamascopium.models.protocols.EncoderInitializable]: Encoder initialization from decoder (`init_encoder_with_decoder_transpose`)
    """

    specs: type[TensorSpecs] = TensorSpecs
    """Tensor specs class for inferring dimension names from tensors. Override in subclasses for custom specs."""

    def __init__(self, cfg: SparseDictionaryConfig, device_mesh: Optional[DeviceMesh] = None):
        super(SparseDictionary, self).__init__()
        self.cfg = cfg

        # should be set by Trainer during training
        self.current_k = cfg.top_k

        # if cfg.norm_activation == "dataset-wise", the dataset average activation norm should be
        # calculated by the initializer before training starts and set by standardize_parameters_of_dataset_activation_scaling
        self.dataset_average_activation_norm: dict[str, float] | None = None

        self.device_mesh: DeviceMesh | None = device_mesh

        self.activation_function: Callable[[torch.Tensor], torch.Tensor] = self.activation_function_factory(device_mesh)

    @torch.no_grad()
    def set_dataset_average_activation_norm(self, dataset_average_activation_norm: dict[str, float]):
        """Set the dataset average activation norm for training or inference.
        dataset_average_activation_norm is set by the Initializer and only used when cfg.norm_activation == "dataset-wise".
        """
        self.dataset_average_activation_norm = dataset_average_activation_norm

    @torch.no_grad()
    def set_current_k(self, current_k: int):
        """Set the current k for topk activation function.
        This should be set by the Trainer during training.
        """
        self.current_k = current_k

    @torch.no_grad()
    def full_state_dict(self):  # should be overridden by subclasses
        state_dict = self.state_dict()

        # Add dataset_average_activation_norm to state dict
        if self.dataset_average_activation_norm is not None:
            for hook_point, value in self.dataset_average_activation_norm.items():
                state_dict[f"dataset_average_activation_norm.{hook_point}"] = torch.tensor(
                    value, device=self.cfg.device, dtype=self.cfg.dtype
                )
        else:
            for hook_point in self.cfg.associated_hook_points:
                state_dict[f"dataset_average_activation_norm.{hook_point}"] = torch.empty(
                    (), device=self.cfg.device, dtype=self.cfg.dtype
                )

        return cast(dict[str, torch.Tensor], state_dict)

    @torch.no_grad()
    def save_checkpoint(self, ckpt_path: Path | str) -> None:
        # TODO: save the config to MongoDB
        """
        {
            "name": sae_name,
            "config": sae_config,
            "path": final_ckpt_path,
        }
        """
        state_dict = self.full_state_dict()
        if Path(ckpt_path).suffix == ".safetensors":
            state_dict = {k: v.full_tensor() if isinstance(v, DTensor) else v for k, v in state_dict.items()}
            if self.device_mesh is None or is_primary_rank(self.device_mesh):
                safe.save_file(state_dict, ckpt_path, {"version": version("llamascopium")})
        elif Path(ckpt_path).suffix == ".pt":
            state_dict = {k: v.full_tensor() if isinstance(v, DTensor) else v for k, v in state_dict.items()}
            if self.device_mesh is None or is_primary_rank(self.device_mesh):
                torch.save({"sae": state_dict, "version": version("llamascopium")}, ckpt_path)
        elif Path(ckpt_path).suffix == ".dcp":
            fs_writer = FileSystemWriter(ckpt_path)
            assert self.device_mesh is not None, "device_mesh must be provided when saving to DCP checkpoint"
            dcp.save(
                state_dict,
                storage_writer=fs_writer,
                # process_group=get_process_group(self.device_mesh),
                # TODO: Fix checkpoint saving during sweeps. Currently fails due to a bug in dcp.save
                # See: https://github.com/pytorch/pytorch/issues/152310
                # Attempted fix was unsuccessful - waiting for upstream resolution
            )
        else:
            raise ValueError(
                f"Invalid checkpoint path {ckpt_path}. Currently only supports .safetensors and .pt formats."
            )

    @torch.no_grad()
    def save_pretrained(
        self,
        save_path: Path | str,
        fold_activation_scale: bool = False,
    ) -> None:
        os.makedirs(Path(save_path), exist_ok=True)
        if fold_activation_scale:
            assert isinstance(self, DatasetNormStandardizable), (
                f"{type(self).__name__} does not support dataset norm standardization (DatasetNormStandardizable)"
            )
            self.standardize_parameters_of_dataset_norm()

        if self.device_mesh is None:
            self.save_checkpoint(Path(save_path) / "sae_weights.safetensors")
        else:
            self.save_checkpoint(Path(save_path) / "sae_weights.dcp")

    @overload
    def encode(
        self,
        x: Union[
            Float[torch.Tensor, "batch d_model"],
            Float[torch.Tensor, "batch seq_len d_model"],
        ],
        return_hidden_pre: Literal[False] = False,
        **kwargs,
    ) -> Union[Float[torch.Tensor, "batch d_sae"], Float[torch.Tensor, "batch seq_len d_sae"]]: ...

    @overload
    def encode(
        self,
        x: Union[
            Float[torch.Tensor, "batch d_model"],
            Float[torch.Tensor, "batch seq_len d_model"],
        ],
        return_hidden_pre: Literal[True],
        **kwargs,
    ) -> tuple[
        Union[
            Float[torch.Tensor, "batch d_sae"],
            Float[torch.Tensor, "batch seq_len d_sae"],
        ],
        Union[
            Float[torch.Tensor, "batch d_sae"],
            Float[torch.Tensor, "batch seq_len d_sae"],
        ],
    ]: ...

    @abstractmethod
    def encode(
        self,
        x: Union[
            Float[torch.Tensor, "batch d_model"],
            Float[torch.Tensor, "batch seq_len d_model"],
        ],
        return_hidden_pre: bool = False,
        **kwargs,
    ) -> Union[
        Float[torch.Tensor, "batch d_sae"],
        Float[torch.Tensor, "batch seq_len d_sae"],
        tuple[
            Union[
                Float[torch.Tensor, "batch d_sae"],
                Float[torch.Tensor, "batch seq_len d_sae"],
            ],
            Union[
                Float[torch.Tensor, "batch d_sae"],
                Float[torch.Tensor, "batch seq_len d_sae"],
            ],
        ],
    ]:
        """Encode input tensor through the sparse dictionary.
        Ensure that the input activations are normalized by calling [`normalize_activations`][llamascopium.models.sparse_dictionary.SparseDictionary.normalize_activations] before calling this method.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def decode(
        self,
        feature_acts: Union[
            Float[torch.Tensor, "batch d_sae"],
            Float[torch.Tensor, "batch seq_len d_sae"],
        ],
        **kwargs,
    ) -> Union[
        Float[torch.Tensor, "batch d_model"],
        Float[torch.Tensor, "batch seq_len d_model"],
    ]:
        """Decode feature activations to reconstructed input."""
        raise NotImplementedError("Subclasses must implement this method")

    @timer.time("forward")
    def forward(
        self,
        x: Union[
            Float[torch.Tensor, "batch d_model"],
            Float[torch.Tensor, "batch seq_len d_model"],
        ],
        encoder_kwargs: dict[str, Any] = {},
        decoder_kwargs: dict[str, Any] = {},
    ) -> Union[
        Float[torch.Tensor, "batch d_model"],
        Float[torch.Tensor, "batch seq_len d_model"],
    ]:
        """Forward pass through the sparse dictionary.
        Ensure that the input activations are normalized by calling [`normalize_activations`][llamascopium.models.sparse_dictionary.SparseDictionary.normalize_activations] before calling this method.
        """
        feature_acts = self.encode(x, **encoder_kwargs)
        reconstructed = self.decode(feature_acts, **decoder_kwargs)
        return reconstructed

    def compute_norm_factor(self, x: torch.Tensor, hook_point: str) -> torch.Tensor:
        """Compute the normalization factor for the activation vectors.
        This should be called during forward pass.
        There are four modes for norm_activation:
        - "token-wise": normalize by the token-wise norm calculated with the input x
        - "batch-wise": normalize by the batch-wise norm calculated with the input x
        - "dataset-wise": normalize by the dataset-wise norm initialized by the Initializer with sae.set_dataset_average_activation_norm method
        - "inference": no normalization, the weights are already normalized by the Initializer with sae.standardize_parameters_of_dataset_norm method
        """
        assert self.cfg.norm_activation in [
            "token-wise",
            "batch-wise",
            "dataset-wise",
            "inference",
        ], f"Not implemented norm_activation {self.cfg.norm_activation}"
        if self.cfg.norm_activation == "token-wise":
            return math.sqrt(self.cfg.d_model) / torch.norm(x, 2, dim=-1, keepdim=True)
        if self.cfg.norm_activation == "batch-wise":
            return math.sqrt(self.cfg.d_model) / torch.norm(x, 2, dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
        if self.cfg.norm_activation == "dataset-wise":
            assert self.dataset_average_activation_norm is not None, (
                "dataset_average_activation_norm must be provided from Initializer"
            )
            return torch.tensor(
                math.sqrt(self.cfg.d_model) / self.dataset_average_activation_norm[hook_point],
                device=x.device,
                dtype=x.dtype,
            )
        if self.cfg.norm_activation == "inference":
            return torch.tensor(1.0, device=x.device, dtype=x.dtype)
        raise ValueError(f"Not implemented norm_activation {self.cfg.norm_activation}")

    @overload
    def normalize_activations(
        self,
        batch: dict[str, torch.Tensor],
        *,
        return_scale_factor: Literal[False] = False,
    ) -> dict[str, torch.Tensor]: ...

    @overload
    def normalize_activations(
        self, batch: dict[str, torch.Tensor], *, return_scale_factor: Literal[True]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]: ...

    @timer.time("normalize_activations")
    def normalize_activations(
        self, batch: dict[str, torch.Tensor], *, return_scale_factor: bool = False
    ) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Normalize the input activations.
        This should be called before calling [`encode`][llamascopium.models.sparse_dictionary.SparseDictionary.encode] or [`compute_loss`][llamascopium.models.sparse_dictionary.SparseDictionary.compute_loss].
        """

        scale_factors = {
            k: self.compute_norm_factor(v, hook_point=k)
            for k, v in batch.items()
            if k in self.cfg.associated_hook_points
        }
        others = {k: v for k, v in batch.items() if k not in self.cfg.associated_hook_points}
        activations = {k: v * scale_factors[k] for k, v in batch.items() if k in self.cfg.associated_hook_points}

        if not return_scale_factor:
            return activations | others
        else:
            return activations | others, scale_factors

    def denormalize_activations(
        self, batch: dict[str, torch.Tensor], scale_factors: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Denormalize the input activations.
        This should be called after calling [`encode`][llamascopium.models.sparse_dictionary.SparseDictionary.encode] or [`compute_loss`][llamascopium.models.sparse_dictionary.SparseDictionary.compute_loss].
        """
        return {k: v / scale_factors[k] for k, v in batch.items()}

    def get_parameters(self) -> list[dict[str, Any]]:
        """Get the parameters of the model for optimization."""
        jumprelu_params = (
            list(self.activation_function.parameters()) if isinstance(self.activation_function, JumpReLU) else []
        )
        other_params = [p for p in self.parameters() if not any(p is param for param in jumprelu_params)]
        return [
            {"params": other_params, "name": "others"},
            {"params": jumprelu_params, "name": "jumprelu"},
        ]

    def load_full_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True) -> None:
        """Load the full state dictionary into the model. This includes the model weights and the dataset average activation norm."""

        # Extract and set dataset_average_activation_norm if present
        norm_keys = [k for k in state_dict.keys() if k.startswith("dataset_average_activation_norm.")]

        if norm_keys:
            dataset_norm = {key.split(".", 1)[1]: state_dict[key].item() for key in norm_keys}
            self.set_dataset_average_activation_norm(dataset_norm)
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("dataset_average_activation_norm.")}

        self.load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_config(cls, cfg: SparseDictionaryConfig, device_mesh: DeviceMesh | None = None) -> Self:
        """Construct a sparse dictionary from a config. This method is only used to automatically determine the correct registered subclass of SparseDictionary based on the sae_type. Otherwise, it's consistent with directly call the constructor of the subclass."""

        if cls is SparseDictionary:
            cls = SAE_TYPE_TO_MODEL_CLASS[cfg.sae_type]

        if cfg.sae_pretrained_name_or_path is not None:
            return cls.from_pretrained(
                cfg.sae_pretrained_name_or_path,
                device_mesh=device_mesh,
                fold_activation_scale=False,
                strict_loading=cfg.strict_loading,
                device=cfg.device,
                dtype=cfg.dtype,
            )

        model = cls(cfg, device_mesh)
        total_params = sum(param.numel() for param in model.parameters()) / 1e9
        logger.info(f"Initializing {cfg.sae_type} with {total_params:.2f} B parameters")

        return model

    @classmethod
    def from_local(
        cls,
        path: str,
        device_mesh: DeviceMesh | None = None,
        strict_loading: bool = True,
        fold_activation_scale: bool = False,
        **kwargs,
    ):
        """Load a pretrained sparse dictionary from a local directory."""

        cfg = SparseDictionaryConfig.from_pretrained(path, strict_loading=strict_loading, **kwargs)
        cfg.sae_pretrained_name_or_path = None
        model = cls.from_config(cfg, device_mesh=device_mesh)

        if path.endswith(".pt") or path.endswith(".safetensors") or path.endswith(".dcp"):
            ckpt_path = path
        else:
            ckpt_prioritized_paths = [
                f"{path}/sae_weights.safetensors",
                f"{path}/sae_weights.pt",
                f"{path}/sae_weights.dcp",
                f"{path}/checkpoints/pruned.safetensors",
                f"{path}/checkpoints/pruned.pt",
                f"{path}/checkpoints/pruned.dcp",
                f"{path}/checkpoints/final.safetensors",
                f"{path}/checkpoints/final.pt",
                f"{path}/checkpoints/final.dcp",
            ]
            for ckpt_path in ckpt_prioritized_paths:
                if os.path.exists(ckpt_path):
                    break
            else:
                raise FileNotFoundError(f"Pretrained model not found at {path}")

        if ckpt_path.endswith(".safetensors"):
            if device_mesh is None:
                state_dict: dict[str, torch.Tensor] = safe.load_file(ckpt_path, device=cfg.device)
            else:
                with safe_open(ckpt_path, device=int(os.environ["LOCAL_RANK"]), framework="pt") as f:

                    def load_tensor(key: str) -> DTensor:
                        tensor_slice = f.get_slice(key)
                        shape = tensor_slice.get_shape()
                        dim_map = model.dim_maps().get(key, DimMap({}))
                        indices = dim_map.local_slices(shape, device_mesh)
                        local_tensor = tensor_slice[indices].to(model.override_dtypes().get(key, cfg.dtype))
                        return DTensor.from_local(
                            local_tensor,
                            device_mesh=device_mesh,
                            placements=dim_map.placements(device_mesh),
                        )

                    state_dict = {k: load_tensor(k) for k in f.keys()}
        elif ckpt_path.endswith(".pt"):
            state_dict: dict[str, torch.Tensor] = torch.load(
                ckpt_path,
                map_location=cfg.device,
                weights_only=True,
            )["sae"]
        elif ckpt_path.endswith(".dcp"):
            # DCP checkpoint
            fs_reader = FileSystemReader(ckpt_path)
            state_dict = model.full_state_dict()
            dcp.load(state_dict, storage_reader=fs_reader)
        else:
            raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")

        model.load_full_state_dict(state_dict, strict=strict_loading)
        if fold_activation_scale:
            assert isinstance(model, DatasetNormStandardizable), (
                f"{type(model).__name__} does not support dataset norm standardization (DatasetNormStandardizable)"
            )
            model.standardize_parameters_of_dataset_norm()
        return model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str,
        *,
        device_mesh: DeviceMesh | None = None,
        fold_activation_scale: bool = False,
        strict_loading: bool = True,
        **kwargs: Any,
    ) -> Self:
        """Load a pretrained sparse dictionary. This method will automatically determine whether pretrained_name_or_path is a path to a local directory or a name on HuggingFace Hub.

        Args:
            pretrained_name_or_path (str): If loading from local directory, this is the path to the local directory. If loading sparse dictionary from HuggingFace Hub, this is the format <repo_id>:<name>. If loading SAELens compatible format sparse dictionary, this is the format <release>:<sae_id>.
            device_mesh (DeviceMesh | None): The device mesh to use for the model. If None, the model will be loaded on the current device.
            fold_activation_scale (bool): Whether to fold the dataset-wise average activation norm into the weights and biases of the model. See [`standardize_parameters_of_dataset_norm`][llamascopium.models.sparse_dictionary.SparseDictionary.standardize_parameters_of_dataset_norm] method for more details.
            strict_loading (bool): Whether to strictly load the state dictionary. If False, the state dictionary will be loaded with a relaxed strictness, allowing for missing keys or extra keys.
            **kwargs: Additional keyword arguments to pass to the constructor.

        Loading from a local directory:
        ```python
        model = SparseDictionary.from_pretrained("path/to/local/directory")
        ```

        Loading from HuggingFace Hub:
        ```python
        model = SparseDictionary.from_pretrained("org/name:name")
        ```

        Loading SAELens compatible format sparse dictionary:
        ```python
        model = SparseDictionary.from_pretrained("release:sae_id")
        ```
        """

        sae_type = auto_infer_pretrained_sae_type(pretrained_name_or_path)
        if sae_type == PretrainedSAEType.LOCAL:
            return cls.from_local(
                pretrained_name_or_path,
                device_mesh=device_mesh,
                fold_activation_scale=fold_activation_scale,
                strict_loading=strict_loading,
                **kwargs,
            )
        elif sae_type == PretrainedSAEType.HUGGINGFACE:
            repo_id, name = pretrained_name_or_path.split(":")
            return cls.from_hf(
                repo_id,
                name,
                device_mesh=device_mesh,
                fold_activation_scale=fold_activation_scale,
                strict_loading=strict_loading,
                **kwargs,
            )
        elif sae_type == PretrainedSAEType.SAELENS:
            assert device_mesh is None, (
                "Current does not support distributed loading from SAELens. Try converting later to distributed format."
            )

            from sae_lens import SAE

            from llamascopium.models.sae import SparseAutoEncoder

            assert cls is SparseDictionary or cls is SparseAutoEncoder, (
                f"SAELens only supports vanilla SAE architecture, but got {cls.__name__}"
            )

            release, sae_id = pretrained_name_or_path.split(":")
            sae = SAE.from_pretrained(release, sae_id)
            return cast(Self, SparseAutoEncoder.from_saelens(sae, **kwargs))
        else:
            raise ValueError(f"Unsupported pretrained type: {sae_type}")

    @torch.no_grad()
    def log_statistics(self):
        log_dict: dict[str, Any] = {}
        if isinstance(self, NormComputing):
            log_dict["metrics/encoder_norm"] = item(self.encoder_norm().mean())
            log_dict["metrics/decoder_norm"] = item(self.decoder_norm().mean())
            if self.cfg.use_decoder_bias:
                log_dict["metrics/decoder_bias_norm"] = item(self.decoder_bias_norm().mean())
        if "topk" in self.cfg.act_fn:
            log_dict["sparsity/k"] = self.current_k
        if isinstance(self.activation_function, JumpReLU):
            log_dict["metrics/mean_jumprelu_threshold"] = item(
                self.activation_function.log_jumprelu_threshold.exp().mean()
            )
        return log_dict

    def activation_function_factory(
        self, device_mesh: DeviceMesh | None = None
    ) -> Callable[[torch.Tensor], torch.Tensor] | JumpReLU:
        assert self.cfg.act_fn.lower() in [
            "relu",
            "topk",
            "jumprelu",
            "batchtopk",
        ], f"Not implemented activation function {self.cfg.act_fn}"
        if self.cfg.act_fn.lower() == "relu":
            return lambda x: x * x.gt(0).to(x.dtype)
        elif self.cfg.act_fn.lower() == "jumprelu":
            return JumpReLU(
                self.cfg.jumprelu_threshold_window,
                shape=(self.cfg.d_sae,),
                device=self.cfg.device,
                dtype=self.cfg.dtype,
                device_mesh=device_mesh,
            )

        elif self.cfg.act_fn.lower() == "topk":

            def topk_activation(
                x: Union[
                    Float[torch.Tensor, "batch d_sae"],
                    Float[torch.Tensor, "batch seq_len d_sae"],
                ],
            ):
                if self.device_mesh is not None:
                    assert isinstance(x, DTensor)
                    return distributed_topk(
                        x,
                        k=self.current_k,
                        device_mesh=self.device_mesh,
                        dim=-1,
                        mesh_dim_name="model",
                    )
                else:
                    x = x * x.gt(0).to(x.dtype)
                    x = topk(
                        x,
                        k=self.current_k,
                        dim=-1,
                    )
                    return x

            return topk_activation

        elif self.cfg.act_fn.lower() == "batchtopk":

            def batch_topk(
                x: Union[
                    Float[torch.Tensor, "batch d_sae"],
                    Float[torch.Tensor, "batch seq_len d_sae"],
                ],
            ):
                x = x * x.gt(0).to(x.dtype)
                original_shape = None
                if x.ndim == 3:
                    original_shape = (x.size(0), x.size(1))
                    x = x.flatten(end_dim=1)
                result = (
                    distributed_topk(
                        x,
                        k=self.current_k * x.size(0) // mesh_dim_size(x.device_mesh, "data"),
                        device_mesh=x.device_mesh,
                        dim=(-2, -1),
                        mesh_dim_name="model",
                    )
                    if isinstance(x, DTensor)
                    else topk(
                        x,
                        k=self.current_k * x.size(0),
                        dim=(-2, -1),
                        abs=True,
                    )
                )
                if original_shape is not None:
                    result = result.unflatten(dim=0, sizes=original_shape)
                return result

            return batch_topk

        raise ValueError(f"Not implemented activation function {self.cfg.act_fn}")

    @torch.no_grad()
    def init_parameters(self, **kwargs):
        if self.cfg.act_fn.lower() == "jumprelu":
            assert isinstance(self.activation_function, JumpReLU)
            self.activation_function.log_jumprelu_threshold.data.fill_(kwargs["init_log_jumprelu_threshold_value"])

    @overload
    def compute_loss(
        self,
        batch: dict[str, torch.Tensor],
        *,
        sparsity_loss_type: Literal[
            "power", "tanh", "tanh-quad", "jumprelu-l0-quad", "jumprelu-l0-mean-quad", None
        ] = None,
        tanh_stretch_coefficient: float = 4.0,
        p: int = 1,
        target_l0: float | None = None,
        l1_coefficient: float = 1.0,
        lp_coefficient: float = 0.0,
        auxk_coefficient: float = 0.0,
        k_aux: int = 512,
        update_dead_statistics: Callable[[torch.Tensor, torch.Tensor | None, tuple[str, ...]], torch.Tensor]
        | None = None,
        return_aux_data: Literal[True] = True,
        **kwargs,
    ) -> dict[str, Any]: ...

    @overload
    def compute_loss(
        self,
        batch: dict[str, torch.Tensor],
        *,
        sparsity_loss_type: Literal[
            "power", "tanh", "tanh-quad", "jumprelu-l0-quad", "jumprelu-l0-mean-quad", None
        ] = None,
        tanh_stretch_coefficient: float = 4.0,
        p: int = 1,
        target_l0: float | None = None,
        l1_coefficient: float = 1.0,
        lp_coefficient: float = 0.0,
        auxk_coefficient: float = 0.0,
        k_aux: int = 512,
        update_dead_statistics: Callable[[torch.Tensor, torch.Tensor | None, tuple[str, ...]], torch.Tensor]
        | None = None,
        return_aux_data: Literal[False],
        **kwargs,
    ) -> Float[torch.Tensor, " batch"]: ...

    @timer.time("compute_loss")
    def compute_loss(
        self,
        batch: dict[str, torch.Tensor],
        label: (
            Union[
                Float[torch.Tensor, "batch d_model"],
                Float[torch.Tensor, "batch seq_len d_model"],
            ]
            | None
        ) = None,
        *,
        sparsity_loss_type: Literal[
            "power", "tanh", "tanh-quad", "jumprelu-l0-quad", "jumprelu-l0-mean-quad", None
        ] = None,
        tanh_stretch_coefficient: float = 4.0,
        frequency_scale: float = 0.01,
        p: int = 1,
        target_l0: float | None = None,
        l1_coefficient: float = 1.0,
        lp_coefficient: float = 0.0,
        auxk_coefficient: float = 0.0,
        k_aux: int = 512,
        update_dead_statistics: Callable[[torch.Tensor, torch.Tensor | None, tuple[str, ...]], torch.Tensor]
        | None = None,
        return_aux_data: bool = True,
        **kwargs,
    ) -> Union[
        Float[torch.Tensor, " batch"],
        dict[str, Any],
    ]:
        """Compute the loss for the sparse dictionary.
        Ensure that the input activations are normalized by calling [`normalize_activations`][llamascopium.models.sparse_dictionary.SparseDictionary.normalize_activations] before calling this method.
        """
        x, encoder_kwargs, decoder_kwargs = self.prepare_input(batch)

        label = self.prepare_label(batch, **kwargs)

        with timer.time("encode"):
            feature_acts, hidden_pre = self.encode(x, return_hidden_pre=True, **encoder_kwargs)
        with timer.time("decode"):
            reconstructed = self.decode(feature_acts, **decoder_kwargs)

        with timer.time("loss_calculation"):
            l_rec = (reconstructed - label).pow(2).sum(dim=-1)
            l_rec, _ = apply_token_mask(l_rec, self.specs.loss(l_rec), batch.get("mask"), "mean")
            loss_dict: dict[str, Optional[torch.Tensor]] = {
                "l_rec": l_rec,
            }
            loss = l_rec

            if sparsity_loss_type is not None:
                assert isinstance(self, NormComputing), "NormComputing is required for sparsity loss computation"
                with timer.time("sparsity_loss_calculation"):
                    if sparsity_loss_type == "power":
                        l_s = torch.norm(feature_acts * self.decoder_norm(), p=p, dim=-1)
                    elif sparsity_loss_type == "tanh":
                        l_s = torch.tanh(tanh_stretch_coefficient * feature_acts * self.decoder_norm()).sum(dim=-1)
                    elif sparsity_loss_type == "tanh-quad":
                        score = torch.tanh(tanh_stretch_coefficient * feature_acts * self.decoder_norm())

                        # Use local_map to perform mean reduction locally. This will lower the backward memory usage (likely due to DTensor bug).
                        if isinstance(score, DTensor):
                            approx_frequency = cast(
                                torch.Tensor,
                                local_map(
                                    lambda x: einops.reduce(
                                        x,
                                        "... d_sae -> 1 d_sae",
                                        "mean",
                                    ),
                                    DimMap({"model": 1, "data": 0, "head": 0}).placements(score.device_mesh),
                                )(score),
                            ).mean(0)
                        else:
                            approx_frequency = einops.reduce(
                                score,
                                "... d_sae -> d_sae",
                                "mean",
                            )
                        l_s = (approx_frequency * (1 + approx_frequency / frequency_scale)).sum(dim=-1)
                    elif sparsity_loss_type in ("jumprelu-l0-quad", "jumprelu-l0-mean-quad"):
                        assert isinstance(self.activation_function, JumpReLU), (
                            f"{sparsity_loss_type} sparsity loss requires JumpReLU activation"
                        )
                        assert target_l0 is not None and target_l0 > 0, (
                            f"target_l0 must be positive for {sparsity_loss_type} sparsity loss"
                        )
                        hidden_pre_for_l0_loss = (
                            hidden_pre * self.decoder_norm()
                            if self.cfg.sparsity_include_decoder_norm
                            else hidden_pre
                        )
                        l0 = self.activation_function.l0(hidden_pre_for_l0_loss).sum(dim=-1)
                        if sparsity_loss_type == "jumprelu-l0-mean-quad":
                            mean_l0, _ = apply_token_mask(l0, self.specs.loss(l0), batch.get("mask"), "mean")
                            l_s = (2 / target_l0) * (mean_l0 - target_l0).pow(2)
                        else:
                            l_s = (2 / target_l0) * (l0 - target_l0).pow(2)
                    else:
                        raise ValueError(f"sparsity_loss_type f{sparsity_loss_type} not supported.")
                    l_s = l1_coefficient * l_s
                    loss_dict["l_s"] = l_s
                    if l_s.ndim > 0:
                        l_s, _ = apply_token_mask(l_s, self.specs.loss(l_s), batch.get("mask"), "mean")
                    loss = loss + l_s
            else:
                loss_dict["l_s"] = None

            needs_dead_statistics = auxk_coefficient > 0.0 or (
                lp_coefficient > 0.0 and isinstance(self.activation_function, JumpReLU)
            )
            is_dead = None
            n_dead = None
            mean_dead_pre_activation_deficit = None
            if needs_dead_statistics:
                assert update_dead_statistics is not None, (
                    "update_dead_statistics must be set when an auxiliary loss uses dead latents"
                )
                is_dead = update_dead_statistics(
                    feature_acts,
                    batch.get("mask"),
                    self.specs.feature_acts(feature_acts),
                )
                n_dead = is_dead.sum()

            # Dead-latent JumpReLU pre-activation loss. Detaching the threshold
            # prevents the auxiliary objective from reviving features by merely
            # lowering their gates.
            if lp_coefficient > 0.0 and isinstance(self.activation_function, JumpReLU):
                assert isinstance(self, NormComputing), "NormComputing is required for Lp loss computation"
                assert is_dead is not None
                with timer.time("lp_loss_calculation"):
                    jumprelu_threshold = self.activation_function.get_jumprelu_threshold().detach()
                    dead_mask = is_dead.to(hidden_pre.dtype)
                    hidden_pre_for_gate = (
                        hidden_pre * self.decoder_norm()
                        if self.cfg.sparsity_include_decoder_norm
                        else hidden_pre
                    )
                    dead_pre_activation_deficit = (
                        torch.nn.functional.relu(jumprelu_threshold - hidden_pre_for_gate) * dead_mask
                    )
                    mean_dead_pre_activation_deficit = dead_pre_activation_deficit.sum(dim=-1) / max(
                        int(item(n_dead)),
                        1,
                    )
                    mean_dead_pre_activation_deficit, _ = apply_token_mask(
                        mean_dead_pre_activation_deficit,
                        self.specs.loss(mean_dead_pre_activation_deficit),
                        batch.get("mask"),
                        "mean",
                    )
                    l_p = lp_coefficient * dead_pre_activation_deficit.sum(dim=-1)
                    l_p, _ = apply_token_mask(l_p, self.specs.loss(l_p), batch.get("mask"), "mean")
                    loss_dict["l_p"] = l_p
                    loss = loss + l_p
            else:
                loss_dict["l_p"] = None

            # Add AuxK auxiliary loss if enabled
            if auxk_coefficient > 0.0:
                assert is_dead is not None

                with timer.time("auxk_loss_calculation"):
                    # Get reconstruction error
                    e = label - reconstructed  # (batch, d_model) or (batch, seq_len, d_model)

                    # Get the top-k_aux dead latents based on their activation values
                    current_k = self.current_k
                    if isinstance(is_dead, DTensor):
                        self.current_k = min(k_aux, int(item(is_dead.full_tensor().sum())))
                    else:
                        self.current_k = min(k_aux, int(item(is_dead.sum())))

                    if self.current_k > 0:
                        if self.cfg.sparsity_include_decoder_norm:
                            assert isinstance(self, NormComputing), (
                                "NormComputing is required when sparsity_include_decoder_norm is True"
                            )
                            dead_hidden_pre = hidden_pre * is_dead * self.decoder_norm()
                        else:
                            dead_hidden_pre = hidden_pre * is_dead

                        dead_feature_acts = self.activation_function(dead_hidden_pre)

                        if self.cfg.sparsity_include_decoder_norm:
                            assert isinstance(self, NormComputing)
                            dead_feature_acts = dead_feature_acts / self.decoder_norm()
                            dead_hidden_pre = dead_hidden_pre / self.decoder_norm()

                        # Decode auxiliary feature activations
                        aux_reconstructed = dead_feature_acts @ cast(torch.Tensor, self.W_D)

                        if isinstance(aux_reconstructed, DTensor):
                            aux_reconstructed = DimMap({"data": 0}).redistribute(aux_reconstructed)

                        l_aux = (e - aux_reconstructed).pow(2).sum(dim=-1)

                        l_aux = auxk_coefficient * l_aux
                        l_aux, _ = apply_token_mask(l_aux, self.specs.loss(l_aux), batch.get("mask"), "mean")
                        loss_dict["l_aux"] = l_aux
                        loss = loss + l_aux
                    elif auxk_coefficient > 0.0:
                        loss_dict["l_aux"] = torch.zeros_like(l_rec)
                    else:
                        loss_dict["l_aux"] = None

                    self.current_k = current_k

        if return_aux_data:
            return {
                "loss": loss,
                **loss_dict,
                "label": label,
                "mask": batch.get("mask"),
                "n_tokens": batch["tokens"].numel() if batch.get("mask") is None else int(item(batch["mask"].sum())),
                "feature_acts": feature_acts,
                "reconstructed": reconstructed,
                "hidden_pre": hidden_pre,
                "is_dead": is_dead,
                "n_dead": n_dead,
                "dead_pre_activation_deficit": mean_dead_pre_activation_deficit,
                "l1_coefficient": l1_coefficient,
                "lp_coefficient": lp_coefficient,
                "auxk_coefficient": auxk_coefficient,
            }
        return loss

    @abstractmethod
    def prepare_input(
        self, batch: dict[str, torch.Tensor], **kwargs
    ) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
        """Prepare the input for the encoder and decoder.

        Returns:
            tuple: (input_tensor, encoder_kwargs, decoder_kwargs)
                - input_tensor: The input tensor for the encoder
                - encoder_kwargs: Additional arguments for the encoder
                - decoder_kwargs: Additional arguments for the decoder
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def prepare_label(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """Prepare the label for the loss computation."""
        raise NotImplementedError("Subclasses must implement this method")

    @torch.no_grad()
    def compute_training_metrics(
        self,
        **kwargs,
    ) -> dict[str, float]:
        """Compute model-specific training metrics. Logging context is passed as kwargs.

        Returns:
            Dictionary of metric names to values. Should include model-specific metrics
            (e.g., per-layer metrics for CLT, per-head metrics for Crosscoder).
        """
        return {}

    def dim_maps(self) -> dict[str, DimMap]:
        """Return a dictionary mapping parameter names to dimension maps.

        Returns:
            A dictionary mapping parameter names to DimMap objects.
        """
        sae_maps = {}
        if isinstance(self.activation_function, JumpReLU):
            sae_maps.update({f"activation_function.{k}": v for k, v in self.activation_function.dim_maps().items()})
        return sae_maps

    def override_dtypes(self) -> dict[str, torch.dtype]:
        if isinstance(self.activation_function, JumpReLU):
            return {f"activation_function.{k}": v for k, v in self.activation_function.override_dtypes().items()}
        return {}

    def upload_to_hf(
        self,
        repo_id: str,
        name: str,
        lm_config: LanguageModelConfig | None = None,
        *,
        token: Optional[str] = None,
        private: bool = False,
    ) -> None:
        """Upload the sparse dictionary to HuggingFace Hub."""

        # Create a temporary folder to save the sparse dictionary
        tmp_root = execute_and_broadcast(
            lambda: tempfile.mkdtemp(prefix="sae_upload_"),
            self.device_mesh,
        )

        try:
            # Save locally then upload the whole folder
            self.save_pretrained(tmp_root)

            if is_primary_rank(self.device_mesh):
                self.cfg.save_hyperparameters(tmp_root)

                if lm_config is not None:
                    lm_config.save_lm_config(tmp_root)

                create_repo(repo_id=repo_id, private=private, exist_ok=True, token=token)
                upload_folder(
                    folder_path=tmp_root,
                    repo_id=repo_id,
                    path_in_repo=name,
                    commit_message=f"Upload pretrained sparse dictionary {name}",
                    token=token,
                )
        finally:
            if is_primary_rank(self.device_mesh):
                shutil.rmtree(tmp_root, ignore_errors=True)

    @classmethod
    def from_hf(
        cls,
        repo_id: str,
        name: str,
        *,
        device_mesh: DeviceMesh | None = None,
        token: str | None = None,
        strict_loading: bool = True,
        fold_activation_scale: bool = False,
        **kwargs,
    ):
        """Load the sparse dictionary from HuggingFace Hub."""

        allow_patterns = [f"{name}/*"]

        snapshot_path = execute_and_broadcast(
            lambda: snapshot_download(repo_id=repo_id, allow_patterns=allow_patterns, token=token),
            device_mesh,
        )
        local_path = os.path.join(snapshot_path, name)

        if not (Path(local_path) / "config.json").exists():
            raise FileNotFoundError(f"Sparse dictionary {name} does not exist in repo {repo_id}")

        return cls.from_local(
            local_path,
            device_mesh=device_mesh,
            strict_loading=strict_loading,
            fold_activation_scale=fold_activation_scale,
            **kwargs,
        )
