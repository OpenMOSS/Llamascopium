from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Union, Callable, Tuple, ContextManager
from functools import partial
import torch
import torch.nn as nn
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from lm_saes.sae import SparseAutoEncoder
from lm_saes.config import LanguageModelConfig
from lm_saes.resource_loaders import load_model
from lm_saes.lorsa import LowRankSparseAttention
import re
class ReplacementMLP(nn.Module):
    """Wrapper for a TransformerLens MLP layer that adds in extra hooks"""

    def __init__(self, old_mlp: nn.Module):
        super().__init__()
        self.old_mlp = old_mlp
        self.hook_in = HookPoint()
        self.hook_out = HookPoint()

    def forward(self, x):
        x = self.hook_in(x)
        mlp_out = self.old_mlp(x)
        return self.hook_out(mlp_out)

class ReplacementAttention(nn.Module):
    """Wrapper for MultiHeadAttention that adds in extra hooks"""
    
    def __init__(self, old_mha: nn.Module):
        super().__init__()
        self.old_mha = old_mha
        self.hook_in = HookPoint()
        self.hook_out = HookPoint()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hook_in(x)
        # 调用原始的MultiHeadAttention，它现在返回完整的attn_out
        attn_out = self.old_mha(x)
        return self.hook_out(attn_out)

class ReplacementPolicyHead(nn.Module):
    """Wrapper for LC0 PolicyHead that adds in extra hooks"""

    def __init__(self, old_policy_head: nn.Module):
        super().__init__()
        self.old_policy_head = old_policy_head
        self.hook_pre = HookPoint()
        self.hook_post = HookPoint()
        
        # 为Q和K添加hook points
        self.hook_q = HookPoint()
        self.hook_k = HookPoint()

    def forward(self, x):
        x = self.hook_pre(x)
        
        # 手动实现PolicyHead的前向传播，在Q和K处添加hooks
        # Dense1层
        x = self.old_policy_head.dense1(x)
        x = self.old_policy_head.mish(x)
        
        # Q和K投影，并添加hooks
        q = self.old_policy_head.q_proj(x)
        k = self.old_policy_head.k_proj(x)
        
        # 通过hook points
        q = self.hook_q(q)
        k = self.hook_k(k)
        
        # 计算注意力分数
        scores = self.old_policy_head.hook_policy_qk_score(
            torch.matmul(q, k.transpose(-2, -1))
        )
        scores = scores * self.old_policy_head.scale
        
        # Promotion 计算
        promotion_slice = k[:, 56:64, :]
        promotion_out = self.old_policy_head.promotion(promotion_slice)
        
        promotion_out = promotion_out.transpose(1, 2)
        promotion_out_part1, promotion_out_part2 = torch.split(
            promotion_out, [3, 1], dim=1
        )
        promotion_out = torch.add(promotion_out_part1, promotion_out_part2)
        promotion_out = promotion_out.transpose(1, 2)
        promotion_out = promotion_out.reshape(x.shape[0], 1, 24)
        
        promotion_slice2 = scores[:, 48:56, 56:64]
        promotion_out2 = promotion_slice2.reshape(-1, 64, 1)
        promotion_out2 = torch.cat(
            [promotion_out2, promotion_out2, promotion_out2], dim=-1
        )
        promotion_out2 = promotion_out2.reshape(-1, 8, 24)
        
        promotion = promotion_out2 + promotion_out
        promotion = promotion.reshape(-1, 3, 64)
        
        # 组合policy
        policy = torch.cat([scores, promotion], dim=1)
        policy = policy.reshape(-1, 4288)
        
        indices_long = self.old_policy_head.indices.detach().long()
        policy_logits = policy[:, indices_long]
        
        policy_out = self.old_policy_head.hook_policy_out(policy_logits)
        return self.hook_post(policy_out)

class ReplacementModel(HookedTransformer):    
    d_transcoder: int
    transcoders: Dict[int, SparseAutoEncoder]
    lorsas: nn.ModuleList
    mlp_input_hook: str
    mlp_output_hook: str
    original_mlp_output_hook: str
    feature_input_hook: str
    feature_output_hook: str
    attn_input_hook: str
    attn_output_hook: str

    @classmethod
    def from_pretrained_and_transcoders(
        cls,
        model_cfg: LanguageModelConfig,
        transcoders: Dict[int, SparseAutoEncoder],
        lorsas: List[LowRankSparseAttention],
        mlp_input_hook: str = "resid_mid_after_ln",
        mlp_output_hook: str = "hook_mlp_out",
        attn_input_hook: str = "hook_attn_in",
        attn_output_hook: str = "hook_attn_out",
        **kwargs,
    ) -> "ReplacementModel":
        
        print("in replacement model:")
        # f-string debug syntax: use `{var=}` (no spaces before `}`)
        print(f"{mlp_input_hook=}, {mlp_output_hook=}, {attn_input_hook=}, {attn_output_hook=}")
        
        if model_cfg.model_name.endswith(".pt"):
            model_data = torch.load(model_cfg.model_name, map_location="cpu")
            from .utils.convert_leela_weights import convert_leela_weights
            model = convert_leela_weights(model_data, model_cfg)
        else:
            model = super().from_pretrained(
                model_cfg.model_name,
                use_flash_attn=model_cfg.use_flash_attn,
                device=model_cfg.device,
                cache_dir=model_cfg.cache_dir,
                dtype=model_cfg.dtype,
                fold_ln=False,
                center_writing_weights=False,
                center_unembed=False,
            )

        return cls.from_pretrained_model(
            model,
            transcoders,
            lorsas,
            mlp_input_hook,
            mlp_output_hook,
            attn_input_hook,
            attn_output_hook,
        )

    @classmethod
    def from_pretrained_model(
        cls,
        model: HookedTransformer,
        transcoders: Dict[int, SparseAutoEncoder],
        lorsas: List[LowRankSparseAttention],
        mlp_input_hook: str = "resid_mid_after_ln",
        mlp_output_hook: str = "hook_mlp_out",
        attn_input_hook: str = "hook_attn_in",
        attn_output_hook: str = "hook_attn_out",
    ) -> "ReplacementModel":
        
        replacement_model = cls(
            model.cfg,
            tokenizer=None,
            move_to_device=False,
        )
        
        print(f"{replacement_model = }")
        
        original_state_dict = model.state_dict()
        model_state_dict = {}
        
        for key, value in original_state_dict.items():
            if not key.startswith('transcoders.'):
                model_state_dict[key] = value
        
        replacement_model.load_state_dict(model_state_dict, strict=True) # TODO set to true
        
        replacement_model._configure_replacement_model(
            transcoders, lorsas, mlp_input_hook, mlp_output_hook, attn_input_hook, attn_output_hook
        )
        
        replacement_model.to(replacement_model.cfg.device)
        
        return replacement_model

    @classmethod
    def from_pretrained(  # type: ignore[override]
        cls,
        model_cfg: LanguageModelConfig,
        transcoders: Dict[int, SparseAutoEncoder],
        lorsas: List[LowRankSparseAttention],
        mlp_input_hook: str = "mlp.hook_in",
        mlp_output_hook: str = "mlp.hook_out",
        attn_input_hook: str = "attn.hook_in",
        attn_output_hook: str = "attn.hook_out",
        use_lorsa: bool = True,
        **kwargs,
    ) -> "ReplacementModel":
        """Create a ReplacementModel from the name of HookedTransformer and dict of transcoders

        Args:
            model_name (str): the name of the pretrained HookedTransformer that this
                ReplacmentModel will inherit from
            transcoders: (str): Either a predefined transcoder set name, or a config file
                defining where to load them from
            device (torch.device, Optional): the device onto which to load the transcoders
                and HookedTransformer.

        Returns:
            ReplacementModel: The loaded ReplacementModel
        """

        return cls.from_pretrained_and_transcoders(
            model_cfg,
            transcoders,
            lorsas,
            mlp_input_hook=mlp_input_hook,
            mlp_output_hook=mlp_output_hook,
            attn_input_hook=attn_input_hook,
            attn_output_hook=attn_output_hook,
            use_lorsa=use_lorsa,
            **kwargs,
        )


    def _configure_replacement_model(
        self,
        transcoders: Dict[int, SparseAutoEncoder],
        lorsas: List[LowRankSparseAttention],
        mlp_input_hook: str,
        mlp_output_hook: str,
        attn_input_hook: str,
        attn_output_hook: str,
    ):

        for layer_idx, transcoder in transcoders.items():
            transcoder.to(self.cfg.device, self.cfg.dtype)

        if lorsas is not None:
            for lorsa in lorsas:
                assert not lorsa.cfg.skip_bos, "Lorsa must not skip bos, will be handled by replacement model"
                lorsa.to(self.cfg.device, self.cfg.dtype)
        
        self.add_module("transcoders", nn.ModuleDict({str(k): v for k, v in transcoders.items()}))
        self.add_module("lorsas", nn.ModuleList(lorsas))

        self.d_transcoder = list(transcoders.values())[0].cfg.d_sae
        self.d_lorsa = lorsas[0].cfg.d_sae

        self.mlp_input_hook = mlp_input_hook
        self.original_mlp_output_hook = mlp_output_hook
        self.mlp_output_hook = mlp_output_hook + ".hook_out_grad"

        self.attn_input_hook = attn_input_hook
        self.original_attn_output_hook = attn_output_hook
        self.attn_output_hook = attn_output_hook + ".hook_out_grad"

        for block in self.blocks:
            block.mlp = ReplacementMLP(block.mlp)
            block.attn = ReplacementAttention(block.mha)
        
        if hasattr(self, 'policy_head') and self.policy_head is not None:
            self.policy_head = ReplacementPolicyHead(self.policy_head)
            
        self._configure_gradient_flow()
        self.setup()

    def _configure_gradient_flow(self):
        for layer in range(self.cfg.n_layers):
            self._configure_skip_connection(self.blocks[layer], layer)

        def stop_gradient(acts, hook):
            return acts.detach()

        for block in self.blocks:
            if hasattr(block.ln1, 'hook_scale'):
                block.ln1.hook_scale.add_hook(stop_gradient, is_permanent=True)
            if hasattr(block.ln2, 'hook_scale'):
                block.ln2.hook_scale.add_hook(stop_gradient, is_permanent=True)
            # if hasattr(block, "ln1_post") and hasattr(block.ln1_post, 'hook_scale'):
            #     block.ln1_post.hook_scale.add_hook(stop_gradient, is_permanent=True)
            # if hasattr(block, "ln2_post") and hasattr(block.ln2_post, 'hook_scale'):
            #     block.ln2_post.hook_scale.add_hook(stop_gradient, is_permanent=True)
        
        if hasattr(self.policy_head.old_policy_head.mish, 'hook_weight'):
            self.policy_head.old_policy_head.mish.hook_weight.add_hook(
                stop_gradient, is_permanent=True
            )
        # for block in self.blocks:
        #     if hasattr(block, 'hook_attn_pattern'):
        #         block.hook_attn_pattern.add_hook(
        #             stop_gradient, is_permanent=True
        #         )
                
        for param in self.parameters():
            param.requires_grad = False
        
        for name, bias in self._get_requires_grad_bias_params():
            bias.requires_grad_(True)

        def enable_gradient(acts, hook):            
            if acts.is_leaf:
                acts.requires_grad = True
            return acts

        if (hasattr(self, 'policy_head') and 
                hasattr(self.policy_head, 'hook_pre')):
            self.policy_head.hook_pre.add_hook(
                enable_gradient, is_permanent=True
            )

    def _configure_skip_connection(self, block, layer):
        """为指定层配置skip connection - 基于原始replacement_model.py"""
        
        def add_skip_connection(acts: torch.Tensor, hook: HookPoint, grad_hook: HookPoint, replacement_bias: torch.Tensor):
            # 和原始代码一样的逻辑
            assert replacement_bias.requires_grad, "Replacement bias must be a parameter"
            return grad_hook((acts - replacement_bias).detach() + replacement_bias)

        # 为MLP输出添加hook和特殊的grad hook（类似原始代码）
        output_hook_parts = self.original_mlp_output_hook.split(".")
        subblock = block
        for part in output_hook_parts:
            subblock = getattr(subblock, part)
        subblock.hook_out_grad = HookPoint()
        subblock.add_hook(
            partial(
                add_skip_connection,
                grad_hook=subblock.hook_out_grad,
                replacement_bias=self.transcoders[str(layer)].b_D,
            ),
            is_permanent=True,
        )

        # newly added
        # add attn output hook and special grad hook
        output_hook_parts = self.original_attn_output_hook.split(".")
        subblock = block
        for part in output_hook_parts:
            subblock = getattr(subblock, part)
        subblock.hook_out_grad = HookPoint()
        subblock.add_hook(
            partial(
                add_skip_connection,
                grad_hook=subblock.hook_out_grad,
                replacement_bias=self.lorsas[layer].b_D,
            ),
            is_permanent=True,
        )

    def _get_activation_caching_hooks(
        self,
        zero_bos: bool = False,
        sparse: bool = False,
        apply_activation_function: bool = True,
        first_batch_only: bool = False,
    ) -> Tuple[List, List, List[Tuple[str, Callable]]]:

        activation_matrix = [None] * self.cfg.n_layers * 2
        lorsa_attention_pattern = [None] * self.cfg.n_layers
        
        def cache_activations_attn(acts, hook, layer, zero_bos):
            feature_inputs = acts[:1] if first_batch_only else acts
            with torch.no_grad():
                encode_result = self.lorsas[layer].encode(
                    feature_inputs,
                    return_hidden_pre=not apply_activation_function,
                    return_attention_pattern=True
                )

            if not apply_activation_function:
                lorsa_acts = encode_result[1].detach().squeeze(0)
                pattern = encode_result[2].detach().squeeze(0)
            else:
                lorsa_acts = encode_result[0].detach().squeeze(0)
                pattern = encode_result[1].detach().squeeze(0)
            
            if zero_bos:
                lorsa_acts[0] = 0
            if sparse:
                lorsa_acts = lorsa_acts.to_sparse()
                
            activation_matrix[layer] = lorsa_acts
            lorsa_attention_pattern[layer] = pattern
        
        activation_hooks = [
            (
                f"blocks.{layer}.{self.attn_input_hook}",
                partial(cache_activations_attn, layer=layer, zero_bos=zero_bos),
            )
            for layer in range(self.cfg.n_layers)
        ]
        print("init activation_hooks")
        def cache_activations_mlp(acts, hook, layer, zero_bos):
            # 使用individual SAE而不是CrossLayerTranscoder
            feature_inputs = acts[:1] if first_batch_only else acts
            with torch.no_grad():
                transcoder_acts = self.transcoders[str(layer)].encode(
                    feature_inputs,
                    return_hidden_pre=not apply_activation_function
                )

            if not apply_activation_function:
                transcoder_acts = transcoder_acts[1].detach().squeeze(0)
            else:
                transcoder_acts = transcoder_acts.detach().squeeze(0)
            
            if zero_bos:
                transcoder_acts[0] = 0
            if sparse:
                transcoder_acts = transcoder_acts.to_sparse()
            def mlp_offset(layer):
                return self.cfg.n_layers + layer
            
            activation_matrix[mlp_offset(layer)] = transcoder_acts

        activation_hooks.extend([
            (
                f"blocks.{layer}.{self.mlp_input_hook}",
                partial(cache_activations_mlp, layer=layer, zero_bos=zero_bos),
            )
            for layer in range(self.cfg.n_layers)
        ])

        # print(f'{activation_matrix[25] = }')

        return activation_matrix, lorsa_attention_pattern, activation_hooks

    def get_activations(
        self,
        inputs: Union[str, torch.Tensor],
        sparse: bool = False,
        zero_bos: bool = False,
        apply_activation_function: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # 修正返回类型
        """获取transcoder激活 - 类似原始代码但简化版"""
        
        activation_cache, lorsa_attention_pattern, activation_hooks = self._get_activation_caching_hooks(  # 修正解包
            sparse=sparse,
            zero_bos=zero_bos,
            apply_activation_function=apply_activation_function,
        )
        with torch.inference_mode(), self.hooks(activation_hooks):
            logits = self(inputs)
        activation_cache = torch.stack(activation_cache)
        print(f'{activation_cache.shape = }')
        lorsa_attention_pattern = torch.stack(lorsa_attention_pattern)
        if sparse:
            activation_cache = activation_cache.coalesce()
        return logits, activation_cache, lorsa_attention_pattern

    def setup_attribution(
        self,
        inputs: Union[str, torch.Tensor],
        sparse: bool = False,
        zero_bos: bool = False,
        *,
        extra_fwd_hooks: List[Tuple[str, Callable]] | None = None,
        enable_grad: bool = False,
        first_batch_only: bool = False,
    ):

        if isinstance(inputs, torch.Tensor):
            tokens = inputs.squeeze(0) if inputs.shape[0] == 1 else inputs
        else:
            tokens = inputs

        
        activation_matrix, lorsa_attention_pattern, activation_hooks = self._get_activation_caching_hooks(
            sparse=sparse,
            zero_bos=zero_bos,
            first_batch_only=first_batch_only,
        )

        attn_out_cache = {}
        mlp_out_cache = {}
        embed_cache = {}

        def cache_first(cache, acts, hook):
            cache[hook.name] = acts[:1].detach()
            return acts

        first_lane_hooks = [
            (
                f"blocks.{layer}.{self.attn_output_hook}",
                partial(cache_first, attn_out_cache),
            )
            for layer in range(self.cfg.n_layers)
        ]
        first_lane_hooks.extend(
            (
                f"blocks.{layer}.{self.mlp_output_hook}",
                partial(cache_first, mlp_out_cache),
            )
            for layer in range(self.cfg.n_layers)
        )
        first_lane_hooks.append(("hook_embed", partial(cache_first, embed_cache)))

        all_hooks = (
            activation_hooks
            + first_lane_hooks
            + list(extra_fwd_hooks or ())
        )
        grad_context = torch.enable_grad() if enable_grad else torch.no_grad()
        with grad_context:
            logits = self.run_with_hooks(tokens, fwd_hooks=all_hooks)

        lorsa_activation_matrix = activation_matrix[:self.cfg.n_layers]
        tc_activation_matrix = activation_matrix[self.cfg.n_layers:]

        with torch.no_grad():
            lorsa_reconstruction = torch.stack([
                self.lorsas[layer].decode(lorsa_activation_matrix[layer])
                for layer in range(self.cfg.n_layers)
            ])
            if lorsa_reconstruction.ndim == 4:
                lorsa_reconstruction = lorsa_reconstruction.sum(dim=1)

            transcoder_outputs = []
            for layer in range(self.cfg.n_layers):
                layer_act = tc_activation_matrix[layer]
                if isinstance(layer_act, torch.Tensor) and layer_act.layout == torch.sparse_coo:
                    layer_act = layer_act.to_dense()
                transcoder_outputs.append(self.transcoders[str(layer)].decode(layer_act))
            transcoder_reconstruction = torch.stack(transcoder_outputs)

            seq_len = transcoder_reconstruction.shape[-2]
            error_vectors = torch.zeros(
                [self.cfg.n_layers * 2, seq_len, self.cfg.d_model],
                device=self.cfg.device,
                dtype=self.cfg.dtype,
            )
            error_vectors[:self.cfg.n_layers] = torch.cat(
                list(attn_out_cache.values()), dim=0
            ) - lorsa_reconstruction
            error_vectors[self.cfg.n_layers:] = torch.cat(
                list(mlp_out_cache.values()), dim=0
            ) - transcoder_reconstruction

        if zero_bos and isinstance(tokens, torch.Tensor):
            error_vectors[:, 0] = 0
        lorsa_activation_matrix = torch.stack(lorsa_activation_matrix)
        lorsa_attention_pattern = torch.stack(lorsa_attention_pattern)
        tc_activation_matrix = torch.stack(tc_activation_matrix)
        if sparse:
            lorsa_activation_matrix = lorsa_activation_matrix.coalesce()
            tc_activation_matrix = tc_activation_matrix.coalesce()

        token_vectors = embed_cache["hook_embed"].detach()
        if token_vectors.dim() == 3:
            token_vectors = token_vectors[0]

        return logits, lorsa_activation_matrix, lorsa_attention_pattern, tc_activation_matrix, error_vectors, token_vectors

    def setup_intervention_with_freeze(
        self, inputs: Union[str, torch.Tensor], direct_effects: bool = False
    ) -> List[Tuple[str, Callable]]:
        """设置干预和冻结 - 简化版，只处理MLP相关"""
        
        if direct_effects:
            hookpoints_to_freeze = ["hook_pattern", "hook_scale", self.feature_output_hook]
        else:
            hookpoints_to_freeze = ["hook_pattern"]

        freeze_cache, cache_hooks, _ = self.get_caching_hooks(
            names_filter=lambda name: any(hookpoint in name for hookpoint in hookpoints_to_freeze)
        )
        self.run_with_hooks(inputs, fwd_hooks=cache_hooks)

        def freeze_hook(activations, hook):
            cached_values = freeze_cache[hook.name]

            # 处理序列长度不匹配的情况
            if "hook_pattern" in hook.name and activations.shape[2:] != cached_values.shape[2:]:
                new_activations = activations.clone()
                new_activations[:, :, : cached_values.shape[2], : cached_values.shape[3]] = (
                    cached_values
                )
                return new_activations
            elif (
                "hook_scale" in hook.name or self.feature_output_hook in hook.name
            ) and activations.shape[1] != cached_values.shape[1]:
                new_activations = activations.clone()
                new_activations[:, : cached_values.shape[1]] = cached_values
                return new_activations

            assert activations.shape == cached_values.shape, (
                f"Activations shape {activations.shape} does not match cached values"
                f" shape {cached_values.shape} at hook {hook.name}"
            )
            return cached_values

        fwd_hooks = [
            (hookpoint, freeze_hook)
            for hookpoint in freeze_cache.keys()
            if self.feature_input_hook not in hookpoint
        ]
        
        if not direct_effects:
            return fwd_hooks
        return fwd_hooks

    def _get_feature_intervention_hooks(
        self,
        inputs: Union[str, torch.Tensor],
        interventions: List[
            Tuple[int, Union[int, slice, torch.Tensor], int, Union[int, torch.Tensor]]
        ],
        direct_effects: bool = False,
        freeze_attention: bool = True,
        apply_activation_function: bool = True,
    ):
        """获取特征干预hooks - 简化版，只处理MLP"""
        
        from collections import defaultdict
        interventions_by_layer = defaultdict(list)
        for layer, pos, feature_idx, value in interventions:
            interventions_by_layer[layer].append((pos, feature_idx, value))

        # 激活缓存
        activation_cache, lorsa_attention_pattern, activation_hooks = self._get_activation_caching_hooks(
            apply_activation_function=apply_activation_function
        )

        def intervention_hook(activations, hook, layer, layer_interventions):
            transcoder_activations = activation_cache[layer]
            if not apply_activation_function:
                transcoder_activations = (
                    self.transcoders[str(layer)]
                    .activation_function(transcoder_activations.unsqueeze(0))
                    .squeeze(0)
                )
            transcoder_output = self.transcoders[str(layer)].decode(transcoder_activations)
            for pos, feature_idx, value in layer_interventions:
                transcoder_activations[pos, feature_idx] = value
            new_transcoder_output = self.transcoders[str(layer)].decode(transcoder_activations)
            steering_vector = new_transcoder_output - transcoder_output
            return activations + steering_vector

        intervention_hooks = [
            (
                f"blocks.{layer}.{self.feature_output_hook}",
                partial(intervention_hook, layer=layer, layer_interventions=layer_interventions),
            )
            for layer, layer_interventions in interventions_by_layer.items()
        ]

        all_hooks = (
            self.setup_intervention_with_freeze(inputs, direct_effects=direct_effects)
            if direct_effects  # 不需要freeze_attention了
            else []
        )
        all_hooks += activation_hooks + intervention_hooks

        return all_hooks, activation_cache

    @torch.no_grad()
    def feature_intervention(
        self,
        inputs: Union[str, torch.Tensor],
        interventions: List[
            Tuple[int, Union[int, slice, torch.Tensor], int, Union[int, torch.Tensor]]
        ],
        direct_effects: bool = False,
        freeze_attention: bool = True,
        apply_activation_function: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """特征干预 - 简化版，只处理MLP"""
        
        feature_intervention_hook_output = self._get_feature_intervention_hooks(
            inputs,
            interventions,
            direct_effects=direct_effects,
            freeze_attention=freeze_attention,
            apply_activation_function=apply_activation_function,
        )

        hooks, activation_cache = feature_intervention_hook_output

        with self.hooks(hooks):
            logits = self(inputs)

        activation_cache = torch.stack(activation_cache)

        return logits, activation_cache

    # def _get_requires_grad_bias_params(self):
    #     """获取需要梯度的bias参数 - 类似原始代码但简化"""
    #     bias_params = []
    #     for param in self.named_parameters():
    #         if ('.b' in param[0] and 
    #             'transcoders.b_E' not in param[0] and 
    #             'old' not in param[0]):
    #             bias_params.append(param)
    #     return bias_params
    
    def _get_requires_grad_bias_params(self):
        bias_params = []
        for name, p in self.named_parameters():
            if 'old' in name:
                if 'policy_head' in name:
                    pass
                else:
                    continue
            is_bias_like = (name.endswith('.bias') or re.search(r'\.b($|[_\.])', name))
            if not is_bias_like:
                continue
            if re.search(r'^transcoders\.[^.]+\.b_E($|[_\.])', name):
                continue
            if re.search(r'^lorsas\.[^.]+\.b_V($|[_\.])', name):
                continue
            if 'b_Q' in name or 'b_K' in name:
                continue
            if re.search(r'^lorsas\.[^.]+\.smolgen\.[^.]+\.bias($|[_\.])', name):
                continue
            bias_params.append((name, p))
        return bias_params
    
    
    def run_with_replacements(
        self,
        inputs: Union[str, torch.Tensor],
        apply_activation_function: bool = True,
        zero_bos: bool = False,
        enable_grad: bool = False,
        names_filter=None,
        device=None,
        remove_batch_dim: bool = False,
        incl_bwd: bool = False,
        reset_hooks_end: bool = True,
        clear_contexts: bool = False,
        pos_slice=None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Run a forward pass where attention outputs are replaced by LORSA
        decodes and MLP outputs are replaced by Transcoder decodes, while
        also returning a cache of activations like run_with_cache.

        Replacement is performed per-layer by installing forward hooks on
        `attn.hook_in/out` and `mlp.hook_in/out` of each block.

        Args:
            inputs: Input prompt (string or token ids tensor).
            apply_activation_function: If True, use post-activation features for
                encode→decode; otherwise use pre-activation features.
            zero_bos: If True and inputs include a BOS at position 0, zero out
                feature activations at pos 0 before decoding (for both LORSA and
                Transcoder paths).
            enable_grad: If True, run with grad enabled (ignored if incl_bwd=True).
            names_filter, device, remove_batch_dim, incl_bwd, reset_hooks_end,
                clear_contexts, pos_slice: Same semantics as HookedTransformer.run_with_cache.

        Returns:
            Tuple[model_out, cache_dict]: The logits (or model output) and the
            captured activation cache.
        """

        # Capture inputs to attn/MLP per layer to compute replacement outputs at hook_out
        attn_in_cache: Dict[int, torch.Tensor] = {}
        mlp_in_cache: Dict[int, torch.Tensor] = {}

        def capture_attn_in(acts: torch.Tensor, hook: HookPoint, layer: int):
            attn_in_cache[layer] = acts
            return acts

        def replace_attn_out(_acts: torch.Tensor, hook: HookPoint, layer: int):
            # Compute LORSA encode/decode from captured input
            inputs_tensor = attn_in_cache[layer]
            # Encode
            if apply_activation_function:
                lorsa_feats = self.lorsas[layer].encode(inputs_tensor)
            else:
                lorsa_feats = self.lorsas[layer].encode(
                    inputs_tensor, return_hidden_pre=True
                )[1]
            # Optional BOS zeroing
            if zero_bos and lorsa_feats.shape[0] > 0:
                # Keep batch dimension; zero position 0 for all batch elements
                if lorsa_feats.ndim == 3:
                    lorsa_feats[:, 0] = 0
                elif lorsa_feats.ndim == 2:
                    lorsa_feats[0] = 0
            # Decode to attn output space
            replaced = self.lorsas[layer].decode(lorsa_feats)
            return replaced

        def capture_mlp_in(acts: torch.Tensor, hook: HookPoint, layer: int):
            mlp_in_cache[layer] = acts
            return acts

        def replace_mlp_out(_acts: torch.Tensor, hook: HookPoint, layer: int):
            inputs_tensor = mlp_in_cache[layer]
            # Encode single layer via per-layer SAE
            if apply_activation_function:
                transcoder_feats = self.transcoders[str(layer)].encode(
                    inputs_tensor
                )
            else:
                transcoder_feats = self.transcoders[str(layer)].encode(
                    inputs_tensor, return_hidden_pre=True
                )[1]
            if zero_bos:
                if transcoder_feats.ndim == 3:
                    transcoder_feats[:, 0] = 0
                elif transcoder_feats.ndim == 2:
                    transcoder_feats[0] = 0
            # Decode back to model dimension using per-layer transcoder
            replaced = self.transcoders[str(layer)].decode(transcoder_feats)
            return replaced

        # Build replacement hooks for all layers
        replacement_hooks: List[Tuple[str, Callable]] = []
        for layer in range(self.cfg.n_layers):
            replacement_hooks.append(
                (f"blocks.{layer}.attn.hook_in", partial(capture_attn_in, layer=layer))
            )
            replacement_hooks.append(
                (f"blocks.{layer}.attn.hook_out", partial(replace_attn_out, layer=layer))
            )
            replacement_hooks.append(
                (f"blocks.{layer}.mlp.hook_in", partial(capture_mlp_in, layer=layer))
            )
            replacement_hooks.append(
                (f"blocks.{layer}.mlp.hook_out", partial(replace_mlp_out, layer=layer))
            )

        # Compose with caching hooks similar to run_with_cache
        cache_dict, fwd_hooks, bwd_hooks = self.get_caching_hooks(
            names_filter,
            incl_bwd,
            device,
            remove_batch_dim=remove_batch_dim,
            pos_slice=pos_slice,
        )

        # Determine execution context: allow grads if incl_bwd, else follow enable_grad
        context_manager = (
            torch.enable_grad() if (enable_grad or incl_bwd) else torch.inference_mode()
        )

        with context_manager:
            with self.hooks(
                fwd_hooks=fwd_hooks + replacement_hooks,
                bwd_hooks=bwd_hooks,
                reset_hooks_end=reset_hooks_end,
                clear_contexts=clear_contexts,
            ):
                model_out = self(inputs)
                if incl_bwd:
                    model_out.backward()

        return model_out, cache_dict
