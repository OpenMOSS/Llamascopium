import torch
import torch.nn as nn
import copy
from typing import List, Tuple, Optional, Dict, Any
import chess
from lm_saes.circuit.leela_board import LeelaBoard
from pathlib import Path
from safetensors.torch import load_file
import sys

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from src.chess_utils import get_move_from_policy_output_with_prob
except Exception:
    get_move_from_policy_output_with_prob = None


class IntegratedPolicyLens:
    """integrated Policy Lens analyzer, for analyzing the contribution of each layer to the model's output"""
    
    def __init__(self, model):
        """
        initialize Policy Lens
        
        Args:
            model: HookedTransformer model instance
        """
        self.model = model
        self.policy_head = self.model.policy_head
        self.num_layers = len(getattr(self.model, "blocks", [])) or 15
        print(f"Detected {self.num_layers} layers.")
        
        # cache for layernorm copies and alpha
        self.ln_copies: List[Tuple[Any, Any, nn.Parameter, nn.Parameter]] = []
        self.ln_orig: List[Tuple[Any, Any]] = []
        
        # create non-destructive copies of layernorms (with b zeroed) to use in logit-lens
        self.cache_layernorm()
        
        print("Policy Lens initialized successfully!")
        
    def cache_layernorm(self):
        """cache LayerNorm copies for logit lens analysis"""
        blocks = getattr(self.model, "blocks", None)
        if blocks is None:
            raise RuntimeError("Model does not have 'blocks' attribute!")
        
        self.ln_copies = []
        self.ln_orig = []
        
        first_param = next(self.model.parameters())
        target_device = first_param.device
        target_dtype = first_param.dtype
        
        for i, layer in enumerate(blocks):
            ln1 = getattr(layer, "ln1", None)
            ln2 = getattr(layer, "ln2", None)
            alpha_input = getattr(layer, "alpha_input", None)
            alpha_out1 = getattr(layer, "alpha_out1", None)
            
            if ln1 is None or ln2 is None or alpha_input is None or alpha_out1 is None:
                raise RuntimeError(f"Layer {i} missing expected attributes.")
            
            # keep reference to original LN modules (not modified)
            self.ln_orig.append((ln1, ln2))
            
            # create deep copies to be used for post-ln logit-lens transforms
            ln1_copy = copy.deepcopy(ln1)
            ln2_copy = copy.deepcopy(ln2)
    
            ln1_copy.to(device=target_device, dtype=target_dtype)
            ln2_copy.to(device=target_device, dtype=target_dtype)
            
            # zero the bias on the copies only (original model untouched)
            with torch.no_grad():
                ln1_copy.b.data.zero_()
                ln2_copy.b.data.zero_()
            
            # store (ln1_copy, ln2_copy, alpha_input, alpha_out1)
            self.ln_copies.append((ln1_copy, ln2_copy, alpha_input, alpha_out1))
        
        print(f"Created LayerNorm copies for {len(self.ln_copies)} layers (orig model untouched).")
                
    def apply_postln_truncation(self, h: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        given h = resid_post_after_ln at layer_idx (i.e. after ln2 in this layer),
        apply the alpha of the subsequent layers and their LayerNorm copies (b=0) while zeroing out the sublayer outputs.
        return the transformed representation suitable for input to policy_head.
        
        Args:
            h: residual activation
            layer_idx: current layer index
            
        Returns:
            transformed representation
        """
        x = h
        for l in range(layer_idx + 1, self.num_layers):
            ln1_copy, ln2_copy, alpha_in, alpha_out = self.ln_copies[l]
            # multiply by scalar alphas (ensure dtype/device broadcasting)
            # x = x * alpha_in  # attn branch zero + alpha_in * x # no use for BT4
            x = ln1_copy(x)   # apply copy with b=0
            # x = x * alpha_out  # mlp branch zero + alpha_out * x # no use for BT4
            x = ln2_copy(x)
        return x   
    
    def analyze_single_fen(
        self, 
        fen: str, 
        target_move: Optional[str] = None, 
        topk_vocab: int = 2000
    ) -> Dict[str, Any]:
        """
        analyze a single FEN position. if `target_move` (UCI string) is provided,
        calculate its rank/score in each layer.
        
        Args:
            fen: FEN string
            target_move: target move (UCI format, optional)
            topk_vocab: consider how many top logits to search for legal moves (for efficiency)
            
        Returns:
            dictionary containing the analysis results
        """
        print(f"Analyzing FEN: {fen}, target_move: {target_move}")
        
        try:
            board = chess.Board(fen)
            if board.is_game_over():
                print("Warning: Game is over for this position")
        except ValueError as e:
            print(f"Invalid FEN: {e}")
            return {"error": f"Invalid FEN: {str(e)}"}
        
        with torch.no_grad():
            output, cache = self.model.run_with_cache(fen, prepend_bos=False)
        chess_board = chess.Board(fen)
        legal_uci_set = set(move.uci() for move in chess_board.legal_moves)

        def _ensure_policy_2d(t: torch.Tensor) -> torch.Tensor:
            """ensure the tensor shape of the input to get_move_from_policy_output_with_prob is [1, vocab]."""
            if t.dim() == 3:
                # [batch, seq, vocab] -> take the last token
                return t[:, -1, :]
            if t.dim() == 2:
                return t
            if t.dim() == 1:
                return t.unsqueeze(0)
            raise RuntimeError(f"Unexpected policy tensor shape: {tuple(t.shape)}")

        # use the same way as in notebook to get the final layer policy logits: directly use output[0] of run_with_cache
        final_top_legal_moves: list[dict[str, Any]] = []
        final_all_legals: list[tuple[str, float, float]] = []
        uci2idx_fn = None
        try:
            from lm_saes.circuit.leela_board import LeelaBoard as _LeelaBoard
            _lb = _LeelaBoard.from_fen(fen, history_synthesis=True)
            uci2idx_fn = _lb.uci2idx
        except Exception:
            uci2idx_fn = None

        if get_move_from_policy_output_with_prob is not None:
            final_policy_2d = _ensure_policy_2d(output[0])
            final_all_legals = get_move_from_policy_output_with_prob(final_policy_2d, fen, return_list=True)  # type: ignore[assignment]
            if not isinstance(final_all_legals, list):
                final_all_legals = []
        else:
            return {"error": "get_move_from_policy_output_with_prob not available"}

        # pre-build the full information of the final layer: for providing the final layer rank/score/prob for any move
        final_info_by_uci: dict[str, dict[str, float | int]] = {}
        for i, (uci, logit, prob) in enumerate(final_all_legals):
            final_info_by_uci[uci] = {
                "rank": i + 1,
                "score": float(logit),
                "prob": float(prob),
            }

        for uci, logit, prob in final_all_legals[:10]:
            idx_val = int(uci2idx_fn(uci)) if uci2idx_fn is not None else None
            final_top_legal_moves.append(
                {"idx": idx_val, "uci": uci, "score": float(logit), "prob": float(prob)}
            )
        
        # Analyze each layer using Post-LN logit-lens
        layer_analysis = {}
        # select the ground truth of the final layer: the first move in final_top_legal_moves
        final_top_move_uci = final_top_legal_moves[0]['uci'] if len(final_top_legal_moves) > 0 else None
        
        for layer in range(self.num_layers):
            print(f"Analyzing layer {layer}...")
            
            layer_input = cache[f'blocks.{layer}.resid_post_after_ln']
            truncated_rep = self.apply_postln_truncation(layer_input, layer)
            
            # compute logits for this truncated representation
            layer_policy_output = self.policy_head(truncated_rep)
            
            # obtain logits_last (shape [V])
            if layer_policy_output.dim() == 3:
                logits_last = layer_policy_output[0, -1, :]
            elif layer_policy_output.dim() == 2:
                logits_last = layer_policy_output[0, :]
            else:
                raise RuntimeError("Unexpected layer_policy_output shape")

            # keep the same way as in notebook/model output: final layer directly uses output[0] of run_with_cache
            # (to avoid the bias caused by the way policy_head takes the token of resid/implementation details)
            if layer == self.num_layers - 1:
                logits_last = _ensure_policy_2d(output[0])[0]

            # completely reuse the implementation of move.py: directly get the (uci, logit, prob) for all legal moves, and build all subsequent statistics based on it
            current_top_legal_moves: list[dict[str, Any]] = []
            legal_scores: list[tuple[str, float, int]] = []
            probs: list[float] = []
            if get_move_from_policy_output_with_prob is not None:
                layer_policy_2d = _ensure_policy_2d(logits_last)
                all_legals_layer = get_move_from_policy_output_with_prob(layer_policy_2d, fen, return_list=True)
                if isinstance(all_legals_layer, list):
                    for uci, logit, prob in all_legals_layer:
                        idx_int = int(uci2idx_fn(uci)) if uci2idx_fn is not None else -1
                        legal_scores.append((uci, float(logit), idx_int))
                        probs.append(float(prob))
                    for uci, logit, prob in all_legals_layer[:10]:
                        idx_val = int(uci2idx_fn(uci)) if uci2idx_fn is not None else None
                        current_top_legal_moves.append(
                            {"idx": idx_val, "uci": uci, "score": float(logit), "prob": float(prob)}
                        )

            # calculate the logit entropy of the current layer on the legal move set (entropy is smaller when probability is more concentrated)
            def _entropy(p_list):
                import math
                eps = 1e-12
                if not p_list:
                    return None
                return float(-sum(p * math.log(max(p, eps)) for p in p_list))
            logit_entropy = _entropy(probs)

            # compute target rank if requested
            target_info = None
            if target_move is not None:
                if target_move in legal_uci_set:
                    rank = None
                    t_score = None
                    t_prob = None
                    for i, (uci, score_val, idx_val) in enumerate(legal_scores):
                        if uci == target_move:
                            rank = i + 1  # 1-based
                            t_score = score_val
                            t_prob = probs[i] if i < len(probs) else None
                            break
                    target_info = {
                        'uci': target_move,
                        'rank': rank,
                        'score': t_score,
                        'prob': t_prob
                    }
                else:
                    target_info = {
                        'uci': target_move,
                        'rank': None,
                        'score': None,
                        'error': 'Target move is not legal'
                    }

            # calculate the rank/score/prob of the final layer top move in this layer
            final_top_info = None
            if final_top_move_uci is not None:
                f_rank = None
                f_score = None
                f_prob = None
                for i, (uci, score_val, idx_val) in enumerate(legal_scores):
                    if uci == final_top_move_uci:
                        f_rank = i + 1
                        f_score = score_val
                        f_prob = probs[i] if i < len(probs) else None
                        break
                final_top_info = {
                    'uci': final_top_move_uci,
                    'rank': f_rank,
                    'score': f_score,
                    'prob': f_prob,
                }
            
            # Build move rankings
            move_rankings = []
            for move_data in current_top_legal_moves:
                uci = move_data['uci']
                layer_score = move_data['score']
                
                # find final layer rank/score/prob (from the final layer all-legal distribution)
                finfo = final_info_by_uci.get(uci, {})
                final_rank = finfo.get("rank") if isinstance(finfo, dict) else None
                final_score = finfo.get("score") if isinstance(finfo, dict) else None
                final_prob = finfo.get("prob") if isinstance(finfo, dict) else None
                
                move_rankings.append({
                    'move': uci,
                    'layer_score': float(layer_score),
                    'final_rank': final_rank,
                    'final_score': float(final_score) if final_score is not None else None,
                    'final_prob': float(final_prob) if final_prob is not None else None,
                    'rank_change': (final_rank - (len([m for m in current_top_legal_moves if m['score'] > layer_score]) + 1)) if final_rank is not None else None
                })
            
            layer_analysis[f'layer_{layer}'] = {
                'top_legal_moves': current_top_legal_moves,
                'move_rankings': move_rankings,
                'target': target_info,
                'final_top_move': final_top_info,
                'logit_entropy': logit_entropy,
                'legal_scores': legal_scores,  # save the complete score list for KL divergence calculation
                'probs': probs,  # save the probability distribution for KL divergence calculation
            }
        
        # compute the KL divergence and JSD between layers (for detecting phase transitions)
        def _kl_divergence(p_probs, q_probs, p_legal_scores, q_legal_scores):
            """
            calculate the KL divergence KL(P||Q)
            
            Args:
                p_probs: P distribution probability list (corresponding to p_legal_scores)
                q_probs: Q distribution probability list (corresponding to q_legal_scores)
                p_legal_scores: P distribution (uci, score, idx) list
                q_legal_scores: Q distribution (uci, score, idx) list
            
            Returns:
                KL divergence value
            """
            import math
            eps = 1e-12
            
            # build the mapping from uci to probability
            p_dict = {uci: prob for (uci, _, _), prob in zip(p_legal_scores, p_probs)}
            q_dict = {uci: prob for (uci, _, _), prob in zip(q_legal_scores, q_probs)}
            
            # get the union of all legal moves
            all_moves = set(p_dict.keys()) | set(q_dict.keys())
            
            if not all_moves:
                return None
            
            kl_sum = 0.0
            for uci in all_moves:
                p_val = p_dict.get(uci, eps)  # if the move is not in P, use a small value
                q_val = q_dict.get(uci, eps)  # if the move is not in Q, use a small value
                
                # avoid log(0) and division by zero
                if p_val > eps and q_val > eps:
                    kl_sum += p_val * math.log(p_val / q_val)
            
            return float(kl_sum)
        
        def _js_divergence(p_probs, q_probs, p_legal_scores, q_legal_scores):
            """
            calculate the Jensen-Shannon divergence JSD(P||Q)
            
            JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
            where M = 0.5 * (P + Q)
            
            JSD is a true measure (satisfying symmetry, non-negativity, and triangle inequality), ranges between [0, 1] (if using log2) or [0, ln(2)] (if using natural logarithm)
            
            Args:
                p_probs: P distribution probability list (corresponding to p_legal_scores)
                q_probs: Q distribution probability list (corresponding to q_legal_scores)
                p_legal_scores: P distribution (uci, score, idx) list
                q_legal_scores: Q distribution (uci, score, idx) list
            
            Returns:
                JSD value
            """
            import math
            eps = 1e-12
            
            # build the mapping from uci to probability
            p_dict = {uci: prob for (uci, _, _), prob in zip(p_legal_scores, p_probs)}
            q_dict = {uci: prob for (uci, _, _), prob in zip(q_legal_scores, q_probs)}
            
            # get the union of all legal moves
            all_moves = set(p_dict.keys()) | set(q_dict.keys())
            
            if not all_moves:
                return None
            
            # calculate the average distribution M = 0.5 * (P + Q)
            m_dict = {}
            for uci in all_moves:
                p_val = p_dict.get(uci, eps)
                q_val = q_dict.get(uci, eps)
                m_dict[uci] = 0.5 * (p_val + q_val)
            
            # calculate JSD = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
            kl_pm = 0.0
            kl_qm = 0.0
            
            for uci in all_moves:
                p_val = p_dict.get(uci, eps)
                q_val = q_dict.get(uci, eps)
                m_val = m_dict[uci]
                
                # calculate KL(P||M)
                if p_val > eps and m_val > eps:
                    kl_pm += p_val * math.log(p_val / m_val)
                
                # calculate KL(Q||M)
                if q_val > eps and m_val > eps:
                    kl_qm += q_val * math.log(q_val / m_val)
            
            jsd = 0.5 * kl_pm + 0.5 * kl_qm
            return float(jsd)
        
        # compute the KL divergence between adjacent layers
        layer_kl_divergences = []
        for layer in range(1, self.num_layers):
            prev_layer_data = layer_analysis.get(f'layer_{layer-1}')
            curr_layer_data = layer_analysis.get(f'layer_{layer}')
            
            if prev_layer_data and curr_layer_data:
                prev_probs = prev_layer_data.get('probs', [])
                curr_probs = curr_layer_data.get('probs', [])
                prev_legal_scores = prev_layer_data.get('legal_scores', [])
                curr_legal_scores = curr_layer_data.get('legal_scores', [])
                
                if prev_probs and curr_probs:
                    kl_forward = _kl_divergence(
                        curr_probs, prev_probs,
                        curr_legal_scores, prev_legal_scores
                    )  # KL(current layer||previous layer's distribution)
                    kl_backward = _kl_divergence(
                        prev_probs, curr_probs,
                        prev_legal_scores, curr_legal_scores
                    )  # KL(previous layer's distribution||current layer)
                    
                    # calculate the true Jensen-Shannon divergence (JSD)
                    # JSD is a true measure, satisfying symmetry, non-negativity, and triangle inequality
                    jsd = _js_divergence(
                        prev_probs, curr_probs,
                        prev_legal_scores, curr_legal_scores
                    )
                    
                    layer_kl_divergences.append({
                        'from_layer': layer - 1,
                        'to_layer': layer,
                        'kl_forward': kl_forward,  # KL(current layer||previous layer)
                        'kl_backward': kl_backward,  # KL(previous layer||current layer)
                        'jsd': jsd,  # Jensen-Shannon divergence (true measure)
                    })
                else:
                    layer_kl_divergences.append({
                        'from_layer': layer - 1,
                        'to_layer': layer,
                        'kl_forward': None,
                        'kl_backward': None,
                        'jsd': None,
                    })
            else:
                layer_kl_divergences.append({
                    'from_layer': layer - 1,
                    'to_layer': layer,
                    'kl_forward': None,
                    'kl_backward': None,
                    'jsd': None,
                })
        
        # compute the KL divergence of each layer relative to the final layer
        final_layer_data = layer_analysis.get(f'layer_{self.num_layers-1}')
        layer_to_final_kl = []
        if final_layer_data:
            final_probs = final_layer_data.get('probs', [])
            final_legal_scores = final_layer_data.get('legal_scores', [])
            
            for layer in range(self.num_layers):
                layer_data = layer_analysis.get(f'layer_{layer}')
                if layer_data:
                    layer_probs = layer_data.get('probs', [])
                    layer_legal_scores = layer_data.get('legal_scores', [])
                    
                    if layer_probs and final_probs:
                        kl_to_final = _kl_divergence(
                            final_probs, layer_probs,
                            final_legal_scores, layer_legal_scores
                        )  # KL(final layer||current layer)
                        kl_from_final = _kl_divergence(
                            layer_probs, final_probs,
                            layer_legal_scores, final_legal_scores
                        )  # KL(current layer||final layer)
                        jsd = _js_divergence(
                            layer_probs, final_probs,
                            layer_legal_scores, final_legal_scores
                        )  # JSD(current layer||final layer)
                        
                        layer_to_final_kl.append({
                            'layer': layer,
                            'kl_to_final': kl_to_final,
                            'kl_from_final': kl_from_final,
                            'jsd': jsd,  # Jensen-Shannon divergence
                        })
                    else:
                        layer_to_final_kl.append({
                            'layer': layer,
                            'kl_to_final': None,
                            'kl_from_final': None,
                            'jsd': None,
                        })
                else:
                    layer_to_final_kl.append({
                        'layer': layer,
                        'kl_to_final': None,
                        'kl_from_final': None,
                        'jsd': None,
                    })
        
        return {
            'fen': fen,
            'final_layer_predictions': final_top_legal_moves,
            'layer_analysis': layer_analysis,
            'target_move': target_move,
            'num_layers': self.num_layers,
            'final_top_move_uci': final_top_move_uci,
            'layer_kl_divergences': layer_kl_divergences,  # KL divergence between adjacent layers
            'layer_to_final_kl': layer_to_final_kl,  # KL divergence of each layer relative to the final layer
        }
    
    def load_mean_activation(self, hook_point: str, base_dir: str = None) -> Tuple[torch.Tensor, Optional[int]]:
        model_name = self.model.cfg.model_name
        
        if base_dir is None:
            if 'BT4' not in model_name:
                raise ValueError(f"Only BT4 model is supported, received: {model_name}")
            model_prefix = 'BT4'
            
            # determine the subdirectory based on the hook point type
            if 'attn_out' in hook_point:
                subdir = 'attn_out_mean'
            elif 'mlp_out' in hook_point:
                subdir = 'mlp_out_mean'
            else:
                raise ValueError(f"Unknown hook point type: {hook_point}")
            
            base_dir = f"/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N/activations/{model_prefix}/{subdir}"

        safe_name = hook_point.replace(".", "_")
        file_path = Path(base_dir) / f"{safe_name}_mean.safetensors"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Mean activation file does not exist: {file_path}")

        data = load_file(str(file_path))
        mean_activation = data["mean"]
        n_tokens = data["n_tokens"].item() if "n_tokens" in data else None
        
        return mean_activation, n_tokens
    
    def analyze_mean_ablation(
        self, 
        fen: str, 
        hook_types: List[str] = None,
        target_move: Optional[str] = None,
        topk_vocab: int = 2000
    ) -> Dict[str, Any]:
        """
        analyze the mean ablation for each layer
        
        Args:
            fen: FEN string
            hook_types: list of hook types to analyze, e.g. ['attn_out', 'mlp_out'], default is both
            target_move: target move (UCI format, optional)
            topk_vocab: consider how many top logits to search for legal moves
            
        Returns:
            analysis result dictionary
        """
        if hook_types is None:
            hook_types = ['attn_out', 'mlp_out']
        
        print(f"Analyzing Mean Ablation for FEN: {fen}, target_move: {target_move}")
        
        try:
            board = chess.Board(fen)
            if board.is_game_over():
                print("Warning: Game is over for this position")
        except ValueError as e:
            print(f"Invalid FEN: {e}")
            return {"error": f"Invalid FEN: {str(e)}"}
        
        # get the original output
        with torch.no_grad():
            original_output, original_cache = self.model.run_with_cache(fen, prepend_bos=False)
        
        # get the original top move
        lboard = LeelaBoard.from_fen(fen, history_synthesis=True)
        chess_board = chess.Board(fen)
        legal_uci_set = set(move.uci() for move in chess_board.legal_moves)

        def _compute_legal_scores_and_probs(logits_last: torch.Tensor) -> Tuple[List[Tuple[str, float, int]], List[float]]:
            logits_cpu = logits_last.detach().cpu()
            V = logits_cpu.numel()
            idx_to_uci = [lboard.idx2uci(i) for i in range(V)]
            legal_scores_local: List[Tuple[str, float, int]] = []
            for idx_val, uci in enumerate(idx_to_uci):
                if uci in legal_uci_set:
                    legal_scores_local.append((uci, float(logits_cpu[idx_val].item()), idx_val))
            legal_scores_local.sort(key=lambda x: x[1], reverse=True)
            if len(legal_scores_local) == 0:
                return legal_scores_local, []

            if get_move_from_policy_output_with_prob is not None:
                policy_out = logits_cpu.unsqueeze(0)  # [1, vocab]
                all_legals = get_move_from_policy_output_with_prob(policy_out, fen, return_list=True)
                probs_by_uci = {uci: prob for (uci, _logit, prob) in all_legals} if isinstance(all_legals, list) else {}
                probs_local = [float(probs_by_uci.get(uci, 0.0)) for (uci, _, _) in legal_scores_local]
                return legal_scores_local, probs_local

            return legal_scores_local, []

        def _find_move_info(move_uci: str, legal_scores_local: List[Tuple[str, float, int]], probs_local: List[float]) -> Dict[str, Any]:
            if move_uci not in legal_uci_set:
                return {"uci": move_uci, "rank": None, "score": None, "prob": None, "error": "Target move is not legal"}
            for i, (uci, score_val, _) in enumerate(legal_scores_local):
                if uci == move_uci:
                    return {"uci": move_uci, "rank": i + 1, "score": float(score_val), "prob": float(probs_local[i]) if i < len(probs_local) else None}
            return {"uci": move_uci, "rank": None, "score": None, "prob": None, "error": "Target move not found in legal_scores"}
        
        # compute the top move of the original output
        # original_output is a list, output[0] is the logits tensor, shape is [batch, vocab_size]
        original_logits = original_output[0]  # shape: [1, 1858]
        if original_logits.dim() == 2:
            original_logits_last = original_logits[0, :]  # shape: [1858]
        elif original_logits.dim() == 1:
            original_logits_last = original_logits
        else:
            raise RuntimeError(f"Unexpected original logits shape: {original_logits.shape}")
        
        original_vals, original_idxs = torch.sort(original_logits_last, descending=True)
        original_vals = original_vals.detach().cpu().numpy()
        original_idxs = original_idxs.detach().cpu().numpy()
        original_top_legal_moves = []
        
        for idx_val, idx in zip(original_vals, original_idxs):
            uci = lboard.idx2uci(int(idx))
            if uci in legal_uci_set:
                original_top_legal_moves.append({
                    'idx': int(idx),
                    'uci': uci,
                    'score': float(idx_val)
                })
            if len(original_top_legal_moves) >= 10:
                break
        
        original_top_move_uci = original_top_legal_moves[0]['uci'] if len(original_top_legal_moves) > 0 else None

        original_legal_scores, original_probs = _compute_legal_scores_and_probs(original_logits_last)
        original_target_info = _find_move_info(target_move, original_legal_scores, original_probs) if target_move else None
        
        # analyze the mean ablation for each layer and each hook type
        ablation_results = {}
        
        for hook_type in hook_types:
            for layer in range(self.num_layers):
                hook_point = f"blocks.{layer}.hook_{hook_type}"
                print(f"Analyzing {hook_point}...")
                
                try:
                    # load the mean activation
                    mean_activation, n_tokens = self.load_mean_activation(hook_point)
                    mean_activation = mean_activation.to(self.model.cfg.device)
                    
                    # get the shape of the original activation to expand the mean
                    if hook_point in original_cache:
                        original_activation = original_cache[hook_point]
                        batch_size, seq_len = original_activation.shape[:2]
                        mean_expanded = mean_activation.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
                        mean_expanded = mean_expanded.to(original_activation.device)
                    else:
                        print(f"Warning: {hook_point} not in cache, skipping")
                        continue
                    
                    # define the hook function
                    def mean_ablation_hook(activation, hook):
                        return mean_expanded.to(activation.device)
                    
                    # run the ablation
                    with self.model.hooks(fwd_hooks=[(hook_point, mean_ablation_hook)]):
                        with torch.no_grad():
                            ablated_output, _ = self.model.run_with_cache(fen, prepend_bos=False)
                    
                    # get the logits of the ablated output
                    # ablated_output is a list, output[0] is the logits tensor
                    ablated_logits = ablated_output[0]  # shape: [1, 1858]
                    if ablated_logits.dim() == 2:
                        ablated_logits_last = ablated_logits[0, :]  # shape: [1858]
                    elif ablated_logits.dim() == 1:
                        ablated_logits_last = ablated_logits
                    else:
                        raise RuntimeError(f"Unexpected ablated logits shape: {ablated_logits.shape}")
                    
                    # compute the logit difference
                    logit_diff = original_logits_last - ablated_logits_last
                    
                    # get the top move of the ablated output
                    topk = min(topk_vocab, ablated_logits_last.numel())
                    vals, idxs = torch.topk(ablated_logits_last, k=topk)
                    vals_np = vals.detach().cpu().numpy()
                    idxs_np = idxs.detach().cpu().numpy()
                    
                    ablated_top_legal_moves = []
                    for score_val, idx_val in zip(vals_np, idxs_np):
                        uci = lboard.idx2uci(int(idx_val))
                        if uci in legal_uci_set:
                            ablated_top_legal_moves.append({
                                'idx': int(idx_val),
                                'uci': uci,
                                'score': float(score_val)
                            })
                        if len(ablated_top_legal_moves) >= 10:
                            break
                    
                    legal_scores, probs = _compute_legal_scores_and_probs(ablated_logits_last)
                    
                    # compute the entropy
                    def _entropy(p_list):
                        import math
                        eps = 1e-12
                        if not p_list:
                            return None
                        return float(-sum(p * math.log(max(p, eps)) for p in p_list))
                    
                    logit_entropy = _entropy(probs)
                    
                    # target move information
                    target_info = _find_move_info(target_move, legal_scores, probs) if target_move else None

                    target_diff = None
                    if target_move:
                        o = original_target_info or {}
                        a = target_info or {}
                        o_rank = o.get("rank")
                        a_rank = a.get("rank")
                        o_score = o.get("score")
                        a_score = a.get("score")
                        o_prob = o.get("prob")
                        a_prob = a.get("prob")
                        target_diff = {
                            "uci": target_move,
                            "original_rank": o_rank,
                            "ablated_rank": a_rank,
                            "delta_rank": (a_rank - o_rank) if (isinstance(a_rank, int) and isinstance(o_rank, int)) else None,
                            "original_score": o_score,
                            "ablated_score": a_score,
                            "delta_score": (a_score - o_score) if (isinstance(a_score, (int, float)) and isinstance(o_score, (int, float))) else None,
                            "original_prob": o_prob,
                            "ablated_prob": a_prob,
                            "delta_prob": (a_prob - o_prob) if (isinstance(a_prob, (int, float)) and isinstance(o_prob, (int, float))) else None,
                        }
                    
                    # original top move rank in the ablated output
                    original_top_info = None
                    if original_top_move_uci is not None:
                        o_rank = None
                        o_score = None
                        o_prob = None
                        for i, (uci, score_val, idx_val) in enumerate(legal_scores):
                            if uci == original_top_move_uci:
                                o_rank = i + 1
                                o_score = score_val
                                o_prob = probs[i] if i < len(probs) else None
                                break
                        original_top_info = {
                            'uci': original_top_move_uci,
                            'rank': o_rank,
                            'score': o_score,
                            'prob': o_prob
                        }
                    
                    # statistics
                    logit_diff_stats = {
                        'mean': float(logit_diff.mean().item()),
                        'std': float(logit_diff.std().item()),
                        'max': float(logit_diff.max().item()),
                        'min': float(logit_diff.min().item()),
                        'l2_norm': float(torch.norm(logit_diff).item())
                    }
                    
                    ablation_results[hook_point] = {
                        'hook_point': hook_point,
                        'layer': layer,
                        'hook_type': hook_type,
                        'top_legal_moves': ablated_top_legal_moves,
                        'logit_diff_stats': logit_diff_stats,
                        'target': target_info,
                        'target_diff': target_diff,
                        'original_top_move': original_top_info,
                        'logit_entropy': logit_entropy,
                        'n_tokens_in_mean': n_tokens
                    }
                    
                except Exception as e:
                    print(f"Error analyzing {hook_point}: {e}")
                    import traceback
                    traceback.print_exc()
                    ablation_results[hook_point] = {
                        'hook_point': hook_point,
                        'layer': layer,
                        'hook_type': hook_type,
                        'error': str(e)
                    }
        
        return {
            'fen': fen,
            'original_top_legal_moves': original_top_legal_moves,
            'original_top_move_uci': original_top_move_uci,
            'ablation_results': ablation_results,
            'target_move': target_move,
            'original_target': original_target_info,
            'num_layers': self.num_layers,
            'hook_types': hook_types
        }
