from contextlib import contextmanager
from typing import Callable, NamedTuple, cast

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from llamascopium.backend.language_model import TransformerLensLanguageModel
from llamascopium.backend.tl_addons import mount_hooked_modules
from llamascopium.models.lorsa import LowRankSparseAttention
from llamascopium.models.matryoshka_sae import MatryoshkaSparseAutoEncoder
from llamascopium.models.molt import MixtureOfLinearTransform
from llamascopium.models.sae import SparseAutoEncoder
from llamascopium.models.sparse_dictionary import SparseDictionary
from llamascopium.utils.timer import timer


@contextmanager
def apply_saes(
    model: TransformerLensLanguageModel,
    saes: list[SparseDictionary],
    matryoshka_feature_range: tuple[int, int] | None = None,
):
    """
    Apply the sparse dictionaries to the model.
    """
    # Rule out CLTs and crosscoders. Ideally CLTs should be supported.
    assert all(
        [isinstance(sae, SparseAutoEncoder | LowRankSparseAttention | MixtureOfLinearTransform) for sae in saes]
    ), "Currently only support sparse dictionaries that guarantee the hook points in happen before the hook points out."

    saes: list[SparseAutoEncoder | LowRankSparseAttention | MixtureOfLinearTransform] = cast(
        list[SparseAutoEncoder | LowRankSparseAttention | MixtureOfLinearTransform], saes
    )

    fwd_hooks: list[tuple[str | Callable, Callable]] = []
    hook_errors: list[tuple[str, str, HookPoint]] = []

    def get_fwd_hooks(
        sae: SparseAutoEncoder | LowRankSparseAttention | MixtureOfLinearTransform,
    ) -> tuple[list[tuple[str | Callable, Callable]], HookPoint]:
        x = None
        hook_error = HookPoint()

        @timer.time(f"hook_in_{sae.cfg.sae_type}")
        def hook_in(tensor: torch.Tensor, hook: HookPoint):
            nonlocal x
            if (
                x is None
            ):  # Only encode once to prevent re-encoding when a hook is called multiple times, like `blocks.0.ln1.hook_normalized` is called respectively for Q, K and V.
                encoder_kwargs = (
                    {"matryoshka_feature_range": matryoshka_feature_range}
                    if isinstance(sae, MatryoshkaSparseAutoEncoder)
                    else {}
                )
                x = sae.encode(tensor, hook_attn_scores=True, **encoder_kwargs)
            return tensor

        @timer.time(f"hook_out_{sae.cfg.sae_type}")
        def hook_out(tensor: torch.Tensor, hook: HookPoint):
            nonlocal x
            assert x is not None, "hook_in must be called before hook_out."
            reconstructed = sae.decode(x)
            x = None
            return reconstructed + hook_error(tensor - reconstructed)

        return [(sae.cfg.hook_point_in, hook_in), (sae.cfg.hook_point_out, hook_out)], hook_error

    for sae in saes:
        hooks, hook_error = get_fwd_hooks(sae)
        fwd_hooks.extend(hooks)
        hook_errors.append((sae.cfg.hook_point_out, "error", hook_error))

    with mount_hooked_modules(model, [(sae.cfg.hook_point_out, "sae", sae) for sae in saes] + hook_errors):
        with model.hooks(fwd_hooks):
            yield model


@contextmanager
def detach_at(
    model: TransformerLensLanguageModel,
    hook_points: list[str],
):
    """
    Detach the gradients on the given hook points.
    """

    def generate_hook():
        hook_pre = HookPoint()
        hook_post = HookPoint()

        def hook(tensor: torch.Tensor, hook: HookPoint):
            return hook_post(hook_pre(tensor).detach().requires_grad_())

        return hook_pre, hook_post, hook

    hooks = {hook_point: generate_hook() for hook_point in hook_points}
    fwd_hooks: list[tuple[str | Callable, Callable]] = [
        (hook_point, hook) for hook_point, (_, _, hook) in hooks.items()
    ]

    with mount_hooked_modules(
        model,
        [(hook_point, "pre", hook) for hook_point, (hook, _, _) in hooks.items()]
        + [(hook_point, "post", hook) for hook_point, (_, hook, _) in hooks.items()],
    ):
        with model.hooks(fwd_hooks):
            yield model


def _make_bias_leaf(
    bias_data: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device_mesh: DeviceMesh | None = None,
) -> torch.Tensor:
    """Expand a bias tensor to ``(batch_size, seq_len, *bias_shape)`` as a detached leaf.

    When ``device_mesh`` is provided and ``bias_data`` is a plain tensor (e.g. a
    non-sharded model bias like ``b_O``), the leaf is promoted to a replicated
    ``DTensor`` so it composes with DTensor activations inside hooks.
    """
    leaf = bias_data.detach().unsqueeze(0).unsqueeze(0)
    leaf = leaf.expand(batch_size, seq_len, *bias_data.shape).clone()
    if device_mesh is not None and not isinstance(leaf, DTensor):
        leaf = DTensor.from_local(leaf, device_mesh=device_mesh, placements=[Replicate()] * device_mesh.ndim)
    return leaf.requires_grad_(True)


class _BiasReplacer(NamedTuple):
    mount: tuple[str, str, nn.Module]
    fwd_hook: tuple[str | Callable, Callable]
    cache_name: str
    restore: Callable[[], None]


def _make_bias_replacer(
    parent: nn.Module,
    parent_path: str,
    bias_attr: str,
    intercept: str,
    batch_size: int,
    seq_len: int,
    device_mesh: DeviceMesh | None = None,
) -> _BiasReplacer | None:
    param = getattr(parent, bias_attr, None)
    if param is None:
        return None
    original = param.data.clone()
    leaf = _make_bias_leaf(param.data, batch_size, seq_len, device_mesh=device_mesh)
    param.data.zero_()

    bias_hook = HookPoint()
    child_name = f"hook_{bias_attr}"
    return _BiasReplacer(
        mount=(parent_path, child_name, bias_hook),
        fwd_hook=(intercept, lambda tensor, **_: tensor + bias_hook(leaf)),
        cache_name=f"{parent_path}.{child_name}",
        restore=lambda: getattr(parent, bias_attr).data.copy_(original),
    )


@contextmanager
def _bias_phase(model: HookedTransformer, replacers: list[_BiasReplacer | None]):
    active = [r for r in replacers if r is not None]
    if not active:
        yield []
        return
    try:
        with mount_hooked_modules(model, [r.mount for r in active]):
            with model.hooks([r.fwd_hook for r in active]):
                yield [r.cache_name for r in active]
    finally:
        for r in active:
            r.restore()


@contextmanager
def replace_model_biases_with_leaves(
    model: HookedTransformer,
    batch_size: int,
    seq_len: int,
    device_mesh: DeviceMesh | None = None,
):
    """Zero ``attn.b_O`` / ``mlp.b_out`` and expose each as a HookPoint mounted
    on its parent module, with a fwd_hook at ``hook_attn_out`` / ``hook_mlp_out``
    that adds the leaf back.

    **Must be entered BEFORE** :func:`apply_saes` so these fwd_hooks register
    first — a transcoder landing on the same hook point then sees a tensor with
    ``b_O`` / ``b_out`` already restored.

    Mounted hook points:
      - ``blocks.{i}.attn.hook_b_O``    intercept: ``blocks.{i}.hook_attn_out``
      - ``blocks.{i}.mlp.hook_b_out``   intercept: ``blocks.{i}.hook_mlp_out``
    """
    replacers: list[_BiasReplacer | None] = []
    for i, block in enumerate(model.blocks):  # type: ignore[arg-type]
        if hasattr(block, "attn"):
            replacers.append(
                _make_bias_replacer(
                    cast(nn.Module, block.attn),
                    f"blocks.{i}.attn",
                    "b_O",
                    f"blocks.{i}.hook_attn_out",
                    batch_size,
                    seq_len,
                    device_mesh=device_mesh,
                )
            )
        if hasattr(block, "mlp"):
            replacers.append(
                _make_bias_replacer(
                    cast(nn.Module, block.mlp),
                    f"blocks.{i}.mlp",
                    "b_out",
                    f"blocks.{i}.hook_mlp_out",
                    batch_size,
                    seq_len,
                    device_mesh=device_mesh,
                )
            )
    with _bias_phase(model, replacers) as names:
        yield names


@contextmanager
def replace_sae_biases_with_leaves(
    model: HookedTransformer,
    saes: list[SparseDictionary],
    batch_size: int,
    seq_len: int,
    device_mesh: DeviceMesh | None = None,
):
    """Zero SAE / Lorsa biases (``b_D`` / ``b_Q`` / ``b_K``) and expose each as
    a HookPoint mounted on the SAE module, with a fwd_hook at the SAE's own
    ``hook_reconstructed`` / ``hook_q`` / ``hook_k`` that adds the leaf back.

    **Must be entered AFTER** :func:`apply_saes` so the SAE modules are
    reachable in ``model.mod_dict`` at ``{hook_point_out}.sae``.

    Mounted hook points:
      - ``{hook_point_out}.sae.hook_b_D``   intercept: ``{hook_point_out}.sae.hook_reconstructed``
      - ``{hook_point_out}.sae.hook_b_Q``   intercept: ``{hook_point_out}.sae.hook_q``  (Lorsa)
      - ``{hook_point_out}.sae.hook_b_K``   intercept: ``{hook_point_out}.sae.hook_k``  (Lorsa)
    """
    replacers: list[_BiasReplacer | None] = []
    for sae in saes:
        if not isinstance(sae, SparseAutoEncoder | LowRankSparseAttention | MixtureOfLinearTransform):
            continue
        sae_path = f"{sae.cfg.hook_point_out}.sae"
        if isinstance(sae, LowRankSparseAttention):
            replacers.append(
                _make_bias_replacer(
                    sae,
                    sae_path,
                    "b_Q",
                    f"{sae_path}.hook_q",
                    batch_size,
                    seq_len,
                    device_mesh=device_mesh,
                )
            )
            replacers.append(
                _make_bias_replacer(
                    sae,
                    sae_path,
                    "b_K",
                    f"{sae_path}.hook_k",
                    batch_size,
                    seq_len,
                    device_mesh=device_mesh,
                )
            )
        if sae.cfg.use_decoder_bias:
            replacers.append(
                _make_bias_replacer(
                    sae,
                    sae_path,
                    "b_D",
                    f"{sae_path}.hook_reconstructed",
                    batch_size,
                    seq_len,
                    device_mesh=device_mesh,
                )
            )
    with _bias_phase(model, replacers) as names:
        yield names
