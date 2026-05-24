from __future__ import annotations

import json
import os
import re
import warnings
from abc import ABC, abstractmethod
from contextlib import nullcontext
from contextlib import contextmanager
from itertools import accumulate
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Optional,
    Union,
    cast,
)
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.utils._pytree as pytree
from huggingface_hub import hf_hub_download
from torch.distributed import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.experimental import local_map
from transformer_lens import HookedTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BatchFeature,
    Qwen2_5_VLForConditionalGeneration,
)

from llamascopium.backend.evo2 import load_evo2
from llamascopium.backend.tl_addons import run_with_cache_until, run_with_ref_cache
from llamascopium.config import BaseModelConfig
from llamascopium.utils.auto import PretrainedSAEType, auto_infer_pretrained_sae_type
from llamascopium.utils.distributed import DimMap
from llamascopium.utils.misc import pad_and_truncate_tokens
from llamascopium.utils.timer import timer

if TYPE_CHECKING:
    from llamascopium.circuits.indexed_tensor import NodeDimension
    from llamascopium.models.sparse_dictionary import SparseDictionary


def to_tokens(tokenizer, text, max_length, device="cpu", prepend_bos=True):
    tokens = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )["input_ids"].to(device)
    has_bos_prepended = torch.all(tokens[:, 0] == tokenizer.bos_token_id)
    if prepend_bos and not has_bos_prepended:
        tokens = torch.cat(
            [torch.tensor([tokenizer.bos_token_id]).expand(tokens.shape[0]).unsqueeze(-1).to(device), tokens], dim=1
        )
    elif not prepend_bos and has_bos_prepended:
        tokens = tokens[:, 1:]
    return tokens


def set_tokens(tokenizer, bos_token_id, eos_token_id, pad_token_id):
    if tokenizer.eos_token is None:
        if eos_token_id is None:
            tokenizer.eos_token = "<|endoftext|>"
        else:
            tokenizer.eos_token = tokenizer.decode(eos_token_id)
    if tokenizer.pad_token is None:
        if pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.decode(pad_token_id)
    if tokenizer.bos_token is None:
        if bos_token_id is None:
            tokenizer.bos_token = tokenizer.eos_token
        else:
            tokenizer.bos_token = tokenizer.decode(bos_token_id)
    return tokenizer


def _match_str_tokens_to_input(text: str, str_tokens: list[str]) -> list[Optional[dict[str, Any]]]:
    """Match the tokens to the input text, returning a list of tuples of the form (start_idx, end_idx) for each token."""
    # Initialize list to store token positions
    token_positions = []

    # Keep track of current position in text
    curr_pos = 0

    # For each token, try to find its position in the input text
    for token in str_tokens:
        # Optimization: Check if the token appears immediately at the current position
        if text.startswith(token, curr_pos):
            pos = curr_pos
        else:
            # Search for token in remaining text
            pos = text.find(token, curr_pos)

        if pos != -1:
            # Found a match, store position and update curr_pos
            token_positions.append({"key": "text", "range": (pos, pos + len(token))})
            curr_pos = pos + len(token)
        else:
            # No match found. This is only allowed if the token is a special token
            # that doesn't appear in the input text, or if the token is a subword token
            # which cannot be decoded separately.
            # TODO: Deal with subword tokens properly
            if not ((token.startswith("<") and token.endswith(">")) or "�" in token):
                raise ValueError(f"Token {token} not found in input text `{text}`")
            token_positions.append(None)

    return token_positions


def _get_layer_indices_from_hook_points(hook_points: list[str]) -> list[int]:
    residual_pattern = r"^blocks\.(\d+)\.hook_resid_post$"
    matches = [re.match(residual_pattern, hook_point) for hook_point in hook_points]
    assert all(match is not None for match in matches), "hook_points must be residual stream hook points"
    layer_indices = [int(cast(re.Match[str], match).group(1)) for match in matches]
    return layer_indices


class LanguageModelConfig(BaseModelConfig):
    model_name: str = "gpt2"
    """ The name of the model to use. """
    model_from_pretrained_path: str | None = None
    """ The path to the pretrained model. If `None`, will use the model from HuggingFace. """
    cache_dir: str | None = None
    """ The directory of the HuggingFace cache. Should have the same effect as `HF_HOME`. """
    local_files_only: bool = False
    """ Whether to only load the model from the local files. Should have the same effect as `HF_HUB_OFFLINE=1`. """
    max_length: int = 2048
    """ The maximum length of the input. """
    backend: Literal["huggingface", "transformer_lens", "tokenizer_only", "evo2", "auto"] = "auto"
    """ The backend to use for the language model. ``"tokenizer_only"`` loads only the tokenizer without model weights. """
    prepend_bos: bool = True
    """ Whether to prepend the BOS token to the input. """
    bos_token_id: int | None = None
    """ The ID of the BOS token. If `None`, will use the default BOS token. """
    eos_token_id: int | None = None
    """ The ID of the EOS token. If `None`, will use the default EOS token. """
    pad_token_id: int | None = None
    """ The ID of the padding token. If `None`, will use the default padding token. """

    @staticmethod
    def from_pretrained_sae(pretrained_name_or_path: str, **kwargs):
        """Load the LanguageModelConfig from a pretrained SAE name or path. Config is read from <pretrained_name_or_path>/lm_config.json (for local storage), <repo_id>/<name>/lm_config.json (for HuggingFace Hub), or constructed from model name (for SAELens).

        Args:
            pretrained_name_or_path (str): The path to the pretrained SAE.
            **kwargs (Any): Additional keyword arguments to pass to the LanguageModelConfig constructor.
        """
        sae_type = auto_infer_pretrained_sae_type(pretrained_name_or_path.split(":")[0])
        if sae_type == PretrainedSAEType.LOCAL:
            path = os.path.join(os.path.dirname(pretrained_name_or_path), "lm_config.json")
        elif sae_type == PretrainedSAEType.HUGGINGFACE:
            repo_id, name = pretrained_name_or_path.split(":")
            path = hf_hub_download(repo_id=repo_id, filename=f"{name}/lm_config.json")
        elif sae_type == PretrainedSAEType.SAELENS:
            from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory

            repo_id, name = pretrained_name_or_path.split(":")
            lookups = get_pretrained_saes_directory()
            assert lookups.get(repo_id) is not None and lookups[repo_id].saes_map.get(name) is not None, (
                f"Pretrained SAE {pretrained_name_or_path} not found in SAELens. This might indicate bugs in `auto_infer_pretrained_sae_type`."
            )
            model_name = lookups[repo_id].model
            return LanguageModelConfig(model_name=model_name, **kwargs)
        else:
            raise ValueError(f"Unsupported pretrained type: {sae_type}")
        with open(os.path.join(path, "lm_config.json"), "r") as f:
            lm_config = json.load(f)
        return LanguageModelConfig.model_validate(lm_config, **kwargs)

    def save_lm_config(self, sae_path: str):
        assert os.path.exists(sae_path), f"{sae_path} does not exist. Unable to save LanguageModelConfig."

        d = self.model_dump()
        with open(os.path.join(sae_path, "lm_config.json"), "w") as f:
            json.dump(d, f, indent=4)


class LanguageModel(ABC):
    @abstractmethod
    def trace(self, raw: dict[str, Any], n_context: Optional[int] = None) -> list[list[Any]]:
        """Trace how raw data is eventually aligned with tokens.

        Args:
            raw (dict[str, Any]): The raw data to trace.

        Returns:
            list[list[Any]]: The origins of the tokens in the raw data. Shape: (batch_size, n_tokens)
        """
        pass

    @abstractmethod
    def to_activations(
        self, raw: dict[str, Any], hook_points: list[str], n_context: Optional[int] = None
    ) -> dict[str, torch.Tensor]:
        """Convert raw data to activations.

        Args:
            raw (dict[str, Any]): The raw data to convert to activations.
            hook_points (list[str]): The hook points to use for activations.

        Returns:
            dict[str, torch.Tensor]: The activations. Shape: (batch_size, n_tokens, n_activations)
        """
        pass

    @abstractmethod
    def preprocess_raw_data(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Preprocess the raw data.

        Args:
            raw (dict[str, Any]): The raw data to preprocess.

        Returns:
            dict[str, Any]: The preprocessed raw data.
        """
        pass

    @property
    @abstractmethod
    def eos_token_id(self) -> int | None:
        """The ID of the end-of-sequence token."""
        pass

    @property
    @abstractmethod
    def bos_token_id(self) -> int | None:
        """The ID of the beginning-of-sequence token."""
        pass

    @property
    @abstractmethod
    def pad_token_id(self) -> int | None:
        """The ID of the padding token."""
        pass


class TransformerLensLanguageModel(HookedTransformer, LanguageModel):
    """HookedTransformer subclass with distributed (DTensor) support and the LanguageModel interface.

    Use the standard constructor or the convenience class methods:

        # From LanguageModelConfig (loads HF weights internally)
        model = TransformerLensLanguageModel(cfg, device_mesh=mesh)

        # Upgrade an existing HookedTransformer (zero-copy)
        model = TransformerLensLanguageModel.from_hooked_transformer(ht)
    """

    lm_cfg: LanguageModelConfig
    device_mesh: DeviceMesh | None
    device: torch.device

    def __init__(self, cfg: LanguageModelConfig, device_mesh: DeviceMesh | None = None):
        from transformer_lens import loading

        self.lm_cfg = cfg
        self.device_mesh = device_mesh
        if cfg.device == "cuda":
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        elif cfg.device == "npu":
            self.device = torch.device(f"npu:{torch.npu.current_device()}")  # type: ignore[reportAttributeAccessIssue]
        else:
            self.device = torch.device(cfg.device)

        pretrained_path = (
            cfg.model_from_pretrained_path if cfg.model_from_pretrained_path is not None else cfg.model_name
        )

        hf_model = AutoModelForCausalLM.from_pretrained(
            pretrained_path,
            cache_dir=cfg.cache_dir,
            local_files_only=cfg.local_files_only,
            dtype=cfg.dtype,
            trust_remote_code=True,
        )
        hf_tokenizer = AutoTokenizer.from_pretrained(
            pretrained_path,
            cache_dir=cfg.cache_dir,
            trust_remote_code=True,
            use_fast=True,
            add_bos_token=True,
            local_files_only=cfg.local_files_only,
        )

        tl_cfg = loading.get_pretrained_model_config(
            cfg.model_name,
            hf_cfg=hf_model.config.to_dict(),
            device=self.device,
            dtype=cfg.dtype,
        )
        state_dict = loading.get_pretrained_state_dict(cfg.model_name, tl_cfg, hf_model, dtype=cfg.dtype)
        HookedTransformer.__init__(self, tl_cfg, hf_tokenizer, move_to_device=False)
        self.load_and_process_state_dict(state_dict)
        self.move_model_modules_to_device()

        self.tokenizer = set_tokens(hf_tokenizer, cfg.bos_token_id, cfg.eos_token_id, cfg.pad_token_id)

    @classmethod
    def from_hooked_transformer(
        cls,
        model: HookedTransformer,
        device_mesh: DeviceMesh | None = None,
        **overrides: Any,
    ) -> TransformerLensLanguageModel:
        """Upgrade an existing HookedTransformer in-place (zero-copy ``__class__`` swap).

        A ``LanguageModelConfig`` is inferred from the model's
        ``HookedTransformerConfig`` and tokenizer.  Pass keyword *overrides*
        (e.g. ``prepend_bos=False``, ``max_length=512``) for fields that
        cannot be reliably inferred from the architecture.
        """
        model.__class__ = cls
        model = cast(TransformerLensLanguageModel, model)
        tl_cfg = model.cfg
        inferred: dict[str, Any] = dict(
            model_name=tl_cfg.model_name or "unknown",
            max_length=tl_cfg.n_ctx,
            prepend_bos=tl_cfg.default_prepend_bos if tl_cfg.default_prepend_bos is not None else True,
            dtype=tl_cfg.dtype,
            device=str(next(model.parameters()).device),
            bos_token_id=cast(int | None, model.tokenizer.bos_token_id) if model.tokenizer is not None else None,
            eos_token_id=cast(int | None, model.tokenizer.eos_token_id) if model.tokenizer is not None else None,
            pad_token_id=cast(int | None, model.tokenizer.pad_token_id) if model.tokenizer is not None else None,
        )
        model.lm_cfg = LanguageModelConfig(**(inferred | overrides))
        model.device_mesh = device_mesh
        model.device = next(model.parameters()).device
        return model

    def trace(self, raw: dict[str, Any], n_context: Optional[int] = None) -> list[list[Any]]:
        if any(key in ["images", "videos"] for key in raw):
            warnings.warn(
                "Tracing with modalities other than text is not implemented for TransformerLensLanguageModel. Only text fields will be used."
            )
        assert self.tokenizer is not None
        encoding = self.tokenizer(
            raw["text"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.lm_cfg.max_length,
            return_offsets_mapping=True,
        )
        offsets = encoding["offset_mapping"].tolist()  # type: ignore[reportAttributeAccessIssue]
        tokens = cast(torch.Tensor, encoding["input_ids"])  # type: ignore[reportIndexIssue]
        has_bos_prepended = torch.all(tokens[:, 0] == self.bos_token_id)
        if self.lm_cfg.prepend_bos and not has_bos_prepended:
            offsets = [[None] + offset_ for offset_ in offsets]
        elif not self.lm_cfg.prepend_bos and has_bos_prepended:
            offsets = [offset_[1:] for offset_ in offsets]
        if n_context is not None:
            offsets = [offset_[:n_context] for offset_ in offsets]
            offsets = [offset_ + [None] * (n_context - len(offset_)) for offset_ in offsets]
        return [
            [{"key": "text", "range": offset} if offset is not None else None for offset in offset_]
            for offset_ in offsets
        ]

    @torch.no_grad()
    def to_activations(
        self, raw: dict[str, Any], hook_points: list[str], n_context: Optional[int] = None
    ) -> dict[str, torch.Tensor]:
        if any(key in ["images", "videos"] for key in raw):
            warnings.warn(
                "Activations with modalities other than text is not implemented for TransformerLensLanguageModel. Only text fields will be used."
            )
        tokens = to_tokens(
            self.tokenizer,
            raw["text"],
            max_length=self.lm_cfg.max_length,
            device=self.lm_cfg.device,
            prepend_bos=self.lm_cfg.prepend_bos,
        )
        if self.device_mesh is not None and n_context is None:
            num_token_list = [None for _ in range(dist.get_world_size(group=self.device_mesh.get_group("data")))]
            dist.all_gather_object(num_token_list, tokens.shape[1], group=self.device_mesh.get_group("data"))
            n_context = max(cast(list[int], num_token_list))
        if n_context is not None:
            assert self.pad_token_id is not None, (
                "Pad token ID must be set for TransformerLensLanguageModel when n_context is provided"
            )
            tokens = pad_and_truncate_tokens(tokens, n_context, pad_token_id=self.pad_token_id)
        tokens = tokens.contiguous()

        _, activations = self.run_with_cache_until(tokens, names_filter=hook_points, until=hook_points[-1])

        assert self.pad_token_id is not None and self.bos_token_id is not None, "Pad and BOS token IDs must be set"
        mask = torch.logical_and(tokens.ne(self.pad_token_id), tokens.ne(self.bos_token_id)).int()
        attention_mask = torch.logical_and(tokens.ne(self.pad_token_id), tokens.ne(self.bos_token_id)).int()

        return {hook_point: activations[hook_point] for hook_point in hook_points} | {
            "tokens": tokens,
            "mask": mask,
            "attention_mask": attention_mask,
        }

    def preprocess_raw_data(self, raw: dict[str, Any]) -> dict[str, Any]:
        return raw

    @property
    def eos_token_id(self) -> int | None:
        assert self.tokenizer is not None
        return cast(int | None, self.tokenizer.eos_token_id)

    @property
    def bos_token_id(self) -> int | None:
        assert self.tokenizer is not None
        return cast(int | None, self.tokenizer.bos_token_id)

    @property
    def pad_token_id(self) -> int | None:
        assert self.tokenizer is not None
        return cast(int | None, self.tokenizer.pad_token_id)

    # -- device helpers ----------------------------------------------------

    def to(self, device_or_dtype, print_details: bool = True):  # type: ignore[override]
        result = super().to(device_or_dtype, print_details)
        if isinstance(device_or_dtype, (torch.device, str)):
            self.device = torch.device(device_or_dtype)
        return result

    # -- DTensor overrides ------------------------------------------------

    def forward(self, *args, **kwargs):
        if self.device_mesh is not None:
            return local_map(
                super().forward,
                out_placements=DimMap({"data": 0}).placements(self.device_mesh),
            )(*args, prepend_bos=self.lm_cfg.prepend_bos, **kwargs)  # type: ignore
        return super().forward(*args, prepend_bos=self.lm_cfg.prepend_bos, **kwargs)

    @timer.time("to_tensor")
    def _to_tensor(self, input: torch.Tensor) -> torch.Tensor:
        if isinstance(input, DTensor):
            assert input.placements == tuple(DimMap({"data": 0}).placements(cast(DeviceMesh, self.device_mesh)))
            return input.to_local()
        return input

    @timer.time("to_dtensor")
    def _to_dtensor(self, input: torch.Tensor) -> torch.Tensor:
        if isinstance(input, torch.Tensor) and not isinstance(input, DTensor):
            return DTensor.from_local(
                input,
                device_mesh=self.device_mesh,
                placements=DimMap({"data": 0}).placements(cast(DeviceMesh, self.device_mesh)),
            )
        return input

    def _wrap_hook_for_local(self, hook_fn):
        def wrapped_hook_fn(*args, **kwargs):
            if pytree.tree_any(lambda x: isinstance(x, DTensor), args) or pytree.tree_any(
                lambda x: isinstance(x, DTensor), kwargs
            ):
                return hook_fn(*args, **kwargs)
            args = pytree.tree_map(self._to_dtensor, args)
            kwargs = pytree.tree_map(self._to_dtensor, kwargs)
            return pytree.tree_map(self._to_tensor, hook_fn(*args, **kwargs))

        return wrapped_hook_fn

    def run_with_cache(self, *args, **kwargs) -> Any:
        if self.device_mesh is None:
            return super().run_with_cache(*args, **kwargs)

        args = pytree.tree_map(self._to_tensor, args)
        kwargs = pytree.tree_map(self._to_tensor, kwargs)
        return pytree.tree_map(self._to_dtensor, super().run_with_cache(*args, **kwargs))

    def run_with_cache_until(self, *args, **kwargs) -> Any:
        """Run with activation caching, stopping at a given hook for efficiency."""
        if self.device_mesh is None:
            return run_with_cache_until(self, *args, **kwargs)

        args = pytree.tree_map(self._to_tensor, args)
        kwargs = pytree.tree_map(self._to_tensor, kwargs)
        return pytree.tree_map(self._to_dtensor, run_with_cache_until(self, *args, **kwargs))

    @contextmanager  # type: ignore[reportIncompatibleMethodOverride]
    def hooks(
        self,
        fwd_hooks: list[tuple[Union[str, Callable], Callable]] = [],
        bwd_hooks: list[tuple[Union[str, Callable], Callable]] = [],
        reset_hooks_end: bool = True,
        clear_contexts: bool = False,
    ):
        wrapped_fwd_hooks = fwd_hooks
        wrapped_bwd_hooks = bwd_hooks
        if self.device_mesh is not None:
            wrapped_fwd_hooks = [(name, self._wrap_hook_for_local(hook)) for name, hook in fwd_hooks]
            wrapped_bwd_hooks = [(name, self._wrap_hook_for_local(hook)) for name, hook in bwd_hooks]

        with super().hooks(
            fwd_hooks=wrapped_fwd_hooks,
            bwd_hooks=wrapped_bwd_hooks,
            reset_hooks_end=reset_hooks_end,
            clear_contexts=clear_contexts,
        ):
            yield self

    def run_with_ref_cache(self, *args, **kwargs) -> Any:
        if self.device_mesh is None:
            return run_with_ref_cache(self, *args, **kwargs)

        args = pytree.tree_map(self._to_tensor, args)
        kwargs = pytree.tree_map(self._to_tensor, kwargs)
        return pytree.tree_map(self._to_dtensor, run_with_ref_cache(self, *args, **kwargs))

    @contextmanager
    def apply_saes(self, saes: list[SparseDictionary]):
        from llamascopium.circuits.hooks import apply_saes as _apply_saes

        with _apply_saes(self, saes):
            yield self

    @contextmanager
    def detach_at(self, hook_points: list[str]):
        from llamascopium.circuits.hooks import detach_at as _detach_at

        with _detach_at(self, hook_points):
            yield self

    def attribute(
        self,
        inputs: torch.Tensor | str,
        replacement_modules: list[SparseDictionary],
        target_type: Literal["logits", "features"] = "logits",
        features: list[tuple[str, int, int]] | NodeDimension = [],
        max_n_logits: int = 10,
        desired_logit_prob: float = 0.95,
        batch_size: int = 512,
        max_features: int | None = None,
        enable_qk_tracing: bool = False,
        qk_top_fraction: float = 0.6,
        qk_topk: int = 10,
    ):
        from llamascopium.circuits.attribution import attribute

        return attribute(
            self,
            inputs,
            replacement_modules,
            target_type=target_type,
            features=features,
            max_n_logits=max_n_logits,
            desired_logit_prob=desired_logit_prob,
            batch_size=batch_size,
            max_features=max_features,
            enable_qk_tracing=enable_qk_tracing,
            qk_top_fraction=qk_top_fraction,
            qk_topk=qk_topk,
        )

    def qk_trace(
        self,
        inputs: torch.Tensor | str,
        replacement_modules: list[SparseDictionary],
        lorsa_features: NodeDimension,
        topk: int = 10,
        batch_size: int = 1,
    ):
        from llamascopium.circuits.attribution import qk_trace

        return qk_trace(
            self,
            inputs,
            replacement_modules,
            lorsa_features,
            topk,
            batch_size,
        )


class TokenizerOnlyLanguageModel(LanguageModel):
    """Lightweight backend that loads only the tokenizer — no model weights.

    Useful for visualization servers and CLI tools that only need tokenization
    and token-origin tracing.
    """

    def __init__(self, cfg: LanguageModelConfig):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_from_pretrained_path or cfg.model_name,
            cache_dir=cfg.cache_dir,
            trust_remote_code=True,
            use_fast=True,
            add_bos_token=True,
            local_files_only=cfg.local_files_only,
        )
        self.tokenizer = set_tokens(self.tokenizer, cfg.bos_token_id, cfg.eos_token_id, cfg.pad_token_id)

    def trace(self, raw: dict[str, Any], n_context: Optional[int] = None) -> list[list[Any]]:
        encoding = self.tokenizer(
            raw["text"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
            return_offsets_mapping=True,
        )
        offsets = encoding["offset_mapping"].tolist()  # type: ignore[reportAttributeAccessIssue]
        tokens = cast(torch.Tensor, encoding["input_ids"])  # type: ignore[reportIndexIssue]
        has_bos_prepended = torch.all(tokens[:, 0] == self.bos_token_id)
        if self.cfg.prepend_bos and not has_bos_prepended:
            offsets = [[None] + offset_ for offset_ in offsets]
        elif not self.cfg.prepend_bos and has_bos_prepended:
            offsets = [offset_[1:] for offset_ in offsets]
        if n_context is not None:
            offsets = [offset_[:n_context] for offset_ in offsets]
            offsets = [offset_ + [None] * (n_context - len(offset_)) for offset_ in offsets]
        return [
            [{"key": "text", "range": offset} if offset is not None else None for offset in offset_]
            for offset_ in offsets
        ]

    def to_activations(
        self, raw: dict[str, Any], hook_points: list[str], n_context: Optional[int] = None
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError("TokenizerOnlyLanguageModel does not support activation generation")

    def preprocess_raw_data(self, raw: dict[str, Any]) -> dict[str, Any]:
        return raw

    @property
    def eos_token_id(self) -> int | None:
        return cast(int | None, self.tokenizer.eos_token_id)

    @property
    def bos_token_id(self) -> int | None:
        return cast(int | None, self.tokenizer.bos_token_id)

    @property
    def pad_token_id(self) -> int | None:
        return cast(int | None, self.tokenizer.pad_token_id)


class HuggingFaceLanguageModel(LanguageModel):
    def __init__(self, cfg: LanguageModelConfig):
        self.cfg = cfg
        if cfg.device == "cuda":
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        elif cfg.device == "npu":
            self.device = torch.device(f"npu:{torch.npu.current_device()}")  # type: ignore[reportAttributeAccessIssue]
        else:
            self.device = torch.device(cfg.device)

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name if cfg.model_from_pretrained_path is None else cfg.model_from_pretrained_path,
            cache_dir=cfg.cache_dir,
            local_files_only=cfg.local_files_only,
            dtype=cfg.dtype,
            trust_remote_code=True,
        ).to(self.device)  # type: ignore

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name if cfg.model_from_pretrained_path is None else cfg.model_from_pretrained_path,
            cache_dir=cfg.cache_dir,
            trust_remote_code=True,
            use_fast=True,
            add_bos_token=True,
            local_files_only=cfg.local_files_only,
        )
        self.model.eval()

    def preprocess_raw_data(self, raw: dict[str, Any]) -> dict[str, Any]:
        return raw

    @property
    def eos_token_id(self) -> int | None:
        return self.tokenizer.eos_token_id

    @property
    def bos_token_id(self) -> int | None:
        return self.tokenizer.bos_token_id  # should be None

    @property
    def pad_token_id(self) -> int | None:
        return self.tokenizer.pad_token_id

    def to_activations(
        self, raw: dict[str, Any], hook_points: list[str], n_context: Optional[int] = None
    ) -> dict[str, torch.Tensor]:
        layer_indices = _get_layer_indices_from_hook_points(hook_points)
        tokens = to_tokens(
            self.tokenizer,
            raw["text"],
            max_length=self.cfg.max_length,
            device=self.cfg.device,
            prepend_bos=self.cfg.prepend_bos,
        )
        if n_context is not None:
            assert self.pad_token_id is not None, "Pad token ID must be set when n_context is provided"
            tokens = pad_and_truncate_tokens(tokens, n_context, pad_token_id=self.pad_token_id)
        outputs = self.model(tokens, output_hidden_states=True)
        activations = {
            hook_points[i]: outputs.hidden_states[layer_index + 1] for i, layer_index in enumerate(layer_indices)
        }
        activations["tokens"] = tokens
        return activations

    def trace(self, raw: dict[str, Any], n_context: Optional[int] = None) -> list[list[Any]]:
        encoding = self.tokenizer(
            raw["text"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
            return_offsets_mapping=True,
        )
        offsets = encoding["offset_mapping"]
        tokens = encoding["input_ids"]
        has_bos_prepended = torch.all(tokens[:, 0] == self.bos_token_id)
        if self.cfg.prepend_bos and not has_bos_prepended:
            offsets = [[None] + offset_ for offset_ in offsets]
        elif not self.cfg.prepend_bos and has_bos_prepended:
            offsets = [offset_[1:] for offset_ in offsets]
        if n_context is not None:
            offsets = [offset_[:n_context] for offset_ in offsets]
            offsets = [offset_ + [None] * (n_context - len(offset_)) for offset_ in offsets]
        return [[{"key": "text", "range": offset} for offset in offset_] for offset_ in offsets]


class QwenVLLanguageModel(HuggingFaceLanguageModel):
    def __init__(self, cfg: LanguageModelConfig):
        super().__init__(cfg)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            cfg.model_name,
            cache_dir=cfg.cache_dir,
            local_files_only=cfg.local_files_only,
            dtype=cfg.dtype,
        ).to(self.device)  # type: ignore

        self.processor = AutoProcessor.from_pretrained(
            cfg.model_name if cfg.model_from_pretrained_path is None else cfg.model_from_pretrained_path,
            cache_dir=cfg.cache_dir,
            local_files_only=cfg.local_files_only,
            padding_side="left",
            max_pixels=1800 * 28 * 28,
        )
        self.tokenizer = self.processor.tokenizer
        self.model.eval()

    @property
    def eos_token_id(self) -> int | None:
        return self.tokenizer.eos_token_id

    @property
    def bos_token_id(self) -> int | None:
        return self.tokenizer.bos_token_id  # should be None

    @property
    def pad_token_id(self) -> int | None:
        return self.tokenizer.pad_token_id

    def trace(self, raw: dict[str, Any], n_context: Optional[int] = None) -> list[list[Any]]:
        assert self.tokenizer is not None, "tokenizer must be initialized"
        assert self.processor is not None, "processor must be initialized"
        inputs, _ = self.process_raw_data(raw)
        input_ids = inputs["input_ids"]
        if n_context is not None:
            assert self.pad_token_id is not None, (
                "Pad token ID must be set for QwenVLLanguageModel when n_context is provided"
            )
            input_ids = pad_and_truncate_tokens(input_ids, n_context, pad_token_id=self.pad_token_id)
        batch_str_tokens: list[list[str]] = [
            self.tokenizer.batch_decode(input_id, clean_up_tokenization_spaces=False) for input_id in input_ids
        ]

        def split_number(n: int, m: float) -> list[int]:
            # split n into m parts, the parts are as even as possible
            assert m.is_integer()
            quotient = n // int(m)
            remainder = n % int(m)
            return [quotient + 1 if i < remainder else quotient for i in range(int(m))]

        batch_token_positions: list[list[Any]] = [
            _match_str_tokens_to_input(text, str_tokens) for (text, str_tokens) in zip(raw["text"], batch_str_tokens)
        ]

        if "images" in raw and raw["images"] is not None:
            assert "image_grid_thw" in inputs
            assert "pixel_values" in inputs
            resized_shape_list = (inputs["image_grid_thw"][:, 1:] * 14).tolist()
            for str_tokens, images in zip(batch_str_tokens, raw["images"]):
                # str_tokens: list[str], tokens for each text in the batch
                # images: list[torch.Tensor], images for each text in the batch
                # resized_shape: [total_image_number, 2]
                # token_positions: list[Any], positions of the tokens in the input text
                start_id_list = [id for id, str_token in enumerate(str_tokens) if str_token == "<|vision_start|>"]
                end_id_list = [id for id, str_token in enumerate(str_tokens) if str_token == "<|vision_end|>"]
                assert len(start_id_list) == len(end_id_list)
                images_num = len(start_id_list)  # number of images in this text
                resized_shapes = resized_shape_list[:images_num]  # get the resized shapes for images in this text
                resized_shape_list = resized_shape_list[images_num:]

                for i, (start_id, end_id, image, resized_shape) in enumerate(
                    zip(start_id_list, end_id_list, images, resized_shapes)
                ):
                    resized_height, resized_width = int(resized_shape[0]), int(resized_shape[1])
                    original_height, original_width = image.shape[-2], image.shape[-1]
                    image_token_num = end_id - start_id - 1
                    assert image_token_num == resized_height * resized_width / 14 / 14 / 4

                    split_height = split_number(original_height, resized_height / 28)
                    split_width = split_number(original_width, resized_width / 28)

                    prefix_sum_height = [0] + list(accumulate(split_height))
                    prefix_sum_width = [0] + list(accumulate(split_width))

                    prefix_sum_height = [i / original_height for i in prefix_sum_height]
                    prefix_sum_width = [i / original_width for i in prefix_sum_width]
                    grid_coords = [
                        (
                            id // (resized_width // 28),
                            id % (resized_width // 28),
                        )
                        for id in range(image_token_num)
                    ]
                    original_coords = [
                        (
                            prefix_sum_width[grid_coords[id][1]],
                            prefix_sum_height[grid_coords[id][0]],
                            prefix_sum_width[grid_coords[id][1] + 1],
                            prefix_sum_height[grid_coords[id][0] + 1],
                        )
                        for id in range(image_token_num)
                    ]

                    batch_token_positions[i][start_id + 1 : end_id] = original_coords
                    for j in range(len(original_coords)):
                        batch_token_positions[i][start_id + 1 + j] = {
                            "key": "image",
                            "rect": original_coords[j],
                            "image_index": i,
                        }
        return batch_token_positions

    def to_activations(
        self, raw: dict[str, Any], hook_points: list[str], n_context: Optional[int] = None
    ) -> dict[str, torch.Tensor]:
        layer_indices = _get_layer_indices_from_hook_points(hook_points)
        inputs = self.process_raw_data(raw)[0].to(self.device)
        if n_context is not None:
            assert self.pad_token_id is not None, (
                "Pad token ID must be set for QwenVLLanguageModel when n_context is provided"
            )
            inputs["input_ids"] = pad_and_truncate_tokens(
                inputs["input_ids"], n_context, pad_token_id=self.pad_token_id
            )
        outputs = self.model(**inputs, output_hidden_states=True)
        activations = {
            hook_points[i]: outputs.hidden_states[layer_index + 1] for i, layer_index in enumerate(layer_indices)
        }
        activations["tokens"] = inputs["input_ids"]
        return activations

    def process_raw_data(
        self, raw: dict[str, Any], padding: str | bool = "max_length"
    ) -> tuple[BatchFeature, dict[str, Any]]:
        # raw is a dict with keys "text", "images", "videos", etc.
        IMAGE_PAD_TOKEN: str = raw.get("image_pad_token", "<image>")
        inputs = cast(BatchFeature, {})
        processed_raw = {}

        # use process_vision_info to resize images
        assert self.processor is not None, "processor must be initialized"
        if "images" in raw:
            processed_raw["images"] = raw["images"]

        processed_raw["text"] = [
            text.replace(IMAGE_PAD_TOKEN, f"<|vision_start|>{self.processor.image_token}<|vision_end|>")
            for text in raw["text"]
        ]
        inputs: BatchFeature = self.processor(
            text=processed_raw["text"],
            images=raw.get("images", None),
            return_tensors="pt",
            max_length=self.cfg.max_length,
            padding=padding,
            truncation=True,
        )

        return inputs, processed_raw


class Evo2LanguageModel(LanguageModel):
    """Language model wrapper for Evo2 / StripedHyena with explicit hook support."""

    HOOK_PATTERN = re.compile(r"^blocks\.(\d+)\.hook_(mlp|attn|conv|fir|filter)_(in|out)$")

    class _PatchedAttributeHandle:
        def __init__(self, obj: Any, attr: str, replacement: Any):
            self.obj = obj
            self.attr = attr
            self.original = getattr(obj, attr)
            setattr(obj, attr, replacement)

        def remove(self) -> None:
            setattr(self.obj, self.attr, self.original)

    def __init__(self, cfg: LanguageModelConfig):
        self.cfg = cfg
        if cfg.device == "cuda":
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        elif cfg.device == "npu":
            self.device = torch.device(f"npu:{torch.npu.current_device()}")  # type: ignore[reportAttributeAccessIssue]
        else:
            self.device = torch.device(cfg.device)
        self.evo2 = load_evo2(
            model_name=cfg.model_name,
            local_path=cfg.model_from_pretrained_path,
            target_device=self.device,
        )
        self.model = self.evo2.model
        self.model.eval()
        self.tokenizer = self.evo2.tokenizer
        self.primary_device = self.device

    def preprocess_raw_data(self, raw: dict[str, Any]) -> dict[str, Any]:
        return raw

    @property
    def eos_token_id(self) -> int | None:
        return getattr(self.tokenizer, "eos_id", None)

    @property
    def bos_token_id(self) -> int | None:
        return None

    @property
    def pad_token_id(self) -> int | None:
        return getattr(self.tokenizer, "pad_id", None)

    def _tokenize_batch(self, texts: list[str], n_context: Optional[int]) -> torch.Tensor:
        token_lists = [list(self.tokenizer.tokenize(text)) for text in texts]
        max_length = n_context or self.cfg.max_length
        padded = torch.full(
            (len(token_lists), max_length),
            fill_value=self.pad_token_id or 0,
            dtype=torch.long,
            device=self.primary_device,
        )
        for row_idx, token_list in enumerate(token_lists):
            truncated = token_list[:max_length]
            if truncated:
                padded[row_idx, : len(truncated)] = torch.tensor(
                    truncated,
                    dtype=torch.long,
                    device=self.primary_device,
                )
        return padded

    def _resolve_hook_target(self, hook_point: str) -> tuple[int, torch.nn.Module, str, Literal["in", "out"]]:
        match = self.HOOK_PATTERN.match(hook_point)
        if match is None:
            raise ValueError(
                f"Unsupported Evo2 hook point: {hook_point}. "
                "Expected blocks.{layer}.hook_(mlp|attn|conv|fir|filter)_(in|out)."
            )

        layer_idx = int(match.group(1))
        module_kind = match.group(2)
        direction = cast(Literal["in", "out"], match.group(3))
        block = self.model.blocks[layer_idx]

        if module_kind == "mlp":
            module = block.mlp
        elif module_kind == "attn":
            if not hasattr(block, "inner_mha_cls"):
                raise ValueError(f"Layer {layer_idx} is not an Evo2 attention block, cannot capture {hook_point}.")
            module = block.inner_mha_cls
        elif module_kind == "conv":
            module = block
        elif module_kind == "filter":
            if direction != "out":
                raise ValueError(f"Unsupported Evo2 hook point direction for {hook_point}. filter hooks are out-only.")
            if not hasattr(block, "filter"):
                raise ValueError(f"Layer {layer_idx} is not an Evo2 convolution block, cannot capture {hook_point}.")
            module = block.filter
        else:
            if direction != "out":
                raise ValueError(f"Unsupported Evo2 hook point direction for {hook_point}. fir hooks are out-only.")
            if not hasattr(block, "filter") or not hasattr(block.filter, "engine"):
                raise ValueError(f"Layer {layer_idx} is not an Evo2 convolution block, cannot capture {hook_point}.")
            module = block.filter

        return layer_idx, module, module_kind, direction

    @staticmethod
    def _unwrap_activation(tensor_or_tuple: Any) -> torch.Tensor:
        if isinstance(tensor_or_tuple, tuple):
            tensor_or_tuple = tensor_or_tuple[0]
        if not isinstance(tensor_or_tuple, torch.Tensor):
            raise TypeError(f"Expected tensor activation, got {type(tensor_or_tuple).__name__}")
        return tensor_or_tuple

    def to_activations(
        self, raw: dict[str, Any], hook_points: list[str], n_context: Optional[int] = None
    ) -> dict[str, torch.Tensor]:
        texts = raw["text"]
        if isinstance(texts, str):
            texts = [texts]

        tokens = self._tokenize_batch(cast(list[str], texts), n_context)
        activations: dict[str, torch.Tensor] = {}
        handles: list[Any] = []
        fir_capture_seen: set[int] = set()

        for hook_point in hook_points:
            layer_idx, module, module_kind, direction = self._resolve_hook_target(hook_point)

            if module_kind == "fir":
                block = self.model.blocks[layer_idx]
                engine = block.filter.engine

                def _capture_fir_output(output: Any, *, hook_name: str = hook_point) -> None:
                    if layer_idx in fir_capture_seen:
                        return
                    activation = self._unwrap_activation(output).detach().to(self.primary_device)
                    activations[hook_name] = activation
                    fir_capture_seen.add(layer_idx)

                if hasattr(engine, "parallel_fir"):
                    original_parallel_fir = engine.parallel_fir

                    def _parallel_fir_wrapper(*args: Any, **kwargs: Any) -> Any:
                        result = original_parallel_fir(*args, **kwargs)
                        if not kwargs.get("gate", False):
                            _capture_fir_output(result[0] if isinstance(result, tuple) else result)
                        return result

                    handles.append(self._PatchedAttributeHandle(engine, "parallel_fir", _parallel_fir_wrapper))

                if hasattr(engine, "step_fir"):
                    original_step_fir = engine.step_fir

                    def _step_fir_wrapper(*args: Any, **kwargs: Any) -> Any:
                        result = original_step_fir(*args, **kwargs)
                        _capture_fir_output(result[0] if isinstance(result, tuple) else result)
                        return result

                    handles.append(self._PatchedAttributeHandle(engine, "step_fir", _step_fir_wrapper))
                continue

            if module_kind == "conv" and direction == "out":
                def _capture_block_output(
                    _module: torch.nn.Module,
                    _inputs: tuple[torch.Tensor, ...],
                    output: Any,
                    *,
                    hook_name: str = hook_point,
                ) -> None:
                    activation = self._unwrap_activation(output).detach().to(self.primary_device)
                    activations[hook_name] = activation

                handles.append(module.register_forward_hook(_capture_block_output))
                continue

            if direction == "in":
                def _capture_input(
                    _module: torch.nn.Module,
                    inputs: tuple[torch.Tensor, ...],
                    *,
                    hook_name: str = hook_point,
                ) -> None:
                    activation = self._unwrap_activation(inputs[0]).detach().to(self.primary_device)
                    activations[hook_name] = activation

                handles.append(module.register_forward_pre_hook(_capture_input, with_kwargs=False))
            else:
                def _capture_output(
                    _module: torch.nn.Module,
                    _inputs: tuple[torch.Tensor, ...],
                    output: Any,
                    *,
                    hook_name: str = hook_point,
                ) -> None:
                    activation = self._unwrap_activation(output).detach().to(self.primary_device)
                    activations[hook_name] = activation

                handles.append(module.register_forward_hook(_capture_output))

        try:
            with torch.no_grad():
                if torch.cuda.is_available():
                    self.model(tokens)
                else:
                    with patch("torch.cuda.device", new=lambda *_args, **_kwargs: nullcontext()):
                        self.model(tokens)
        finally:
            for handle in handles:
                handle.remove()

        missing = [hook_point for hook_point in hook_points if hook_point not in activations]
        if missing:
            raise RuntimeError(f"Failed to capture Evo2 activations for hook points: {missing}")

        activations["tokens"] = tokens
        activations["mask"] = tokens != (self.pad_token_id or 0)
        activations["attention_mask"] = activations["mask"]
        return activations

    def trace(self, raw: dict[str, Any], n_context: Optional[int] = None) -> list[list[Any]]:
        texts = raw["text"]
        if isinstance(texts, str):
            texts = [texts]

        max_length = n_context or self.cfg.max_length
        traces: list[list[Any]] = []
        for text in cast(list[str], texts):
            token_count = min(len(self.tokenizer.tokenize(text)), max_length)
            offsets = [{"key": "text", "range": (i, i + 1)} for i in range(token_count)]
            offsets.extend([None] * (max_length - token_count))
            traces.append(offsets)
        return traces
