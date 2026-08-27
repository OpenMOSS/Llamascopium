from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Optional, Callable, Any

import sys
import torch
import chess
from lm_saes.circuit.leela_board import LeelaBoard

DEFAULT_MODEL_NAME = "lc0/BT4-1024x15x32h"

# Optional external model getter for sharing cache with app.py
_external_model_getter: Optional[Callable[[str], object]] = None


def set_model_getter(getter: Callable[[str], object]) -> None:
    """Set the external model getter for sharing cached models.

    Args:
        getter: A function that takes model_name and returns a HookedTransformer model.
    """
    global _external_model_getter
    _external_model_getter = getter


def _try_get_from_circuits_service(model_name: str) -> Optional[object]:
    """Try to get the cached model from circuits_service."""
    try:
        import sys
        import os
        server_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if server_dir not in sys.path:
            sys.path.insert(0, server_dir)
        
        from circuits_service import get_cached_models
        cached_model, _, _, _ = get_cached_models(model_name)
        if cached_model is not None:
            print(f"✅ [model_interface] Using cached model from circuits_service: {model_name}")
            return cached_model
    except ImportError:
        pass
    except Exception as e:
        print(f"⚠️ [model_interface] Failed to get model from circuits_service: {e}")
    return None


@lru_cache(maxsize=4)
def _get_model_internal(model_name: str = DEFAULT_MODEL_NAME) -> object:
    """Internal model loader (used only when external getter is not available)."""
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained_no_processing(
        model_name,
        dtype=torch.float32,
    )
    return model.eval()


def _get_model(model_name: str = DEFAULT_MODEL_NAME) -> object:
    """Get model, preferring external cache."""
    if _external_model_getter is not None:
        return _external_model_getter(model_name)
    cached = _try_get_from_circuits_service(model_name)
    if cached is not None:
        return cached
    return _get_model_internal(model_name)


@lru_cache(maxsize=256)
def _run_model_outputs(fen: str, model_name: str) -> tuple[torch.Tensor, ...]:
    if not fen:
        raise ValueError("FEN string should not be empty")
    model = _get_model(model_name)
    with torch.no_grad():
        output, _ = model.run_with_cache(fen, prepend_bos=False)
    if not isinstance(output, Sequence) or len(output) < 3:
        raise RuntimeError("Output format is invalid; expected policy, value, and m.")
    return tuple(output)


def get_policy(fen: str, model_name: str = DEFAULT_MODEL_NAME) -> torch.Tensor:
    outputs = _run_model_outputs(fen, model_name)
    return outputs[0]


def get_value(fen: str, model_name: str = DEFAULT_MODEL_NAME) -> torch.Tensor:
    """Get value output (WDL format).

    Args:
        fen: FEN string.
        model_name: Model name.

    Returns:
        Tensor of shape [1, 3]: [win_rate, draw_rate, lose_rate].
        - value[0][0]: win rate
        - value[0][1]: draw rate
        - value[0][2]: lose rate
    """
    outputs = _run_model_outputs(fen, model_name)
    return outputs[1]


def get_wl(fen: str, model_name: str = DEFAULT_MODEL_NAME) -> float:
    """Get Win-Loss value (wl_ = win_rate - lose_rate).

    Args:
        fen: FEN string.
        model_name: Model name.

    Returns:
        Win-Loss value (typically in [-1, 1]).
    """
    value = get_value(fen, model_name)
    win_rate = value[0][0].item()
    lose_rate = value[0][2].item()
    return win_rate - lose_rate


def get_d(fen: str, model_name: str = DEFAULT_MODEL_NAME) -> float:
    """Get draw probability (d_ = draw_rate).

    Args:
        fen: FEN string.
        model_name: Model name.

    Returns:
        Draw probability in [0, 1].
    """
    value = get_value(fen, model_name)
    return value[0][1].item()


def get_q(fen: str, model_name: str = DEFAULT_MODEL_NAME, draw_score: float = 0.0) -> float:
    """Get Q value: Q = wl + draw_score * d (wl = win_rate - lose_rate, d = draw_rate).

    Args:
        fen: FEN string.
        model_name: Model name.
        draw_score: Draw score (default 0.0).

    Returns:
        Q value (typically near [-1, 1], depending on draw_score).
    """
    wl = get_wl(fen, model_name)
    d = get_d(fen, model_name)
    return wl + draw_score * d


def get_m(fen: str, model_name: str = DEFAULT_MODEL_NAME) -> torch.Tensor:
    """Get moves-left value (m_).

    Args:
        fen: FEN string.
        model_name: Model name.

    Returns:
        Moves-left tensor.
    """
    outputs = _run_model_outputs(fen, model_name)
    return outputs[2]


def policy_tensor_to_move_dict(
    policy_tensor: torch.Tensor,
    fen: str,
    legal_moves: list[chess.Move] | None = None,
) -> dict[str, float]:
    """Convert policy tensor to move dict (move_uci -> probability) using LeelaBoard.

    Args:
        policy_tensor: Policy tensor, shape [1, 1858] or [1858].
        fen: FEN string for LeelaBoard.
        legal_moves: Optional list of legal moves; if None, derived from FEN.

    Returns:
        Dict mapping UCI move strings to probabilities (normalized over legal moves).
    """
    if policy_tensor.dim() > 1:
        policy_logits = policy_tensor[0] if policy_tensor.dim() == 2 else policy_tensor
    else:
        policy_logits = policy_tensor
    
    if isinstance(policy_logits, torch.Tensor):
        policy_logits = policy_logits.cpu()
    
    lboard = LeelaBoard.from_fen(fen, history_synthesis=True)
    
    if legal_moves is None:
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)
    
    legal_uci_set = set(move.uci() for move in legal_moves)
    
    if isinstance(policy_logits, torch.Tensor):
        policy_probs = torch.softmax(policy_logits, dim=0)
    else:
        import numpy as np
        policy_probs = torch.softmax(torch.from_numpy(policy_logits), dim=0)
    
    policy_dict: dict[str, float] = {}
    total_prob = 0.0
    
    for idx in range(len(policy_probs)):
        try:
            uci = lboard.idx2uci(idx)
            if uci in legal_uci_set:
                prob = policy_probs[idx].item() if isinstance(policy_probs[idx], torch.Tensor) else float(policy_probs[idx])
                policy_dict[uci] = prob
                total_prob += prob
        except (KeyError, IndexError, ValueError):
            continue
    
    if total_prob > 0:
        policy_dict = {uci: prob / total_prob for uci, prob in policy_dict.items()}
    else:
        uniform_prob = 1.0 / len(legal_moves) if legal_moves else 0.0
        policy_dict = {move.uci(): uniform_prob for move in legal_moves}
    
    return policy_dict


def evaluate_position(fen: str, model_name: str = DEFAULT_MODEL_NAME) -> dict[str, object]:
    """Evaluate a position and return q, d, m, p (for search Backend).

    Args:
        fen: FEN string.
        model_name: Model name.

    Returns:
        Dict with: q (win-loss), d (draw prob), m (moves left), p (policy dict).
    """
    value = get_value(fen, model_name)
    win_rate = value[0][0].item()
    draw_rate = value[0][1].item()
    lose_rate = value[0][2].item()
    
    wl = win_rate - lose_rate
    
    m_tensor = get_m(fen, model_name)
    m_value = m_tensor[0][0].item() if m_tensor.dim() > 1 else m_tensor[0].item()
    
    policy_tensor = get_policy(fen, model_name)
    policy_dict = policy_tensor_to_move_dict(policy_tensor, fen)
    
    return {
        'q': wl,
        'd': draw_rate,
        'm': m_value,
        'p': policy_dict,
    }


def evaluate_position_for_search(fen: str, model_name: str = DEFAULT_MODEL_NAME) -> dict[str, object]:
    """Evaluate position for MCTS search; returns q, d, m, p for SimpleBackend.

    Args:
        fen: FEN string.
        model_name: Model name.

    Returns:
        Dict with q (float), d (float), m (float), p (move_uci -> probability).
    """
    policy_tensor = get_policy(fen, model_name)
    value_tensor = get_value(fen, model_name)
    m_tensor = get_m(fen, model_name)
    
    win_rate = value_tensor[0][0].item()
    draw_rate = value_tensor[0][1].item()
    lose_rate = value_tensor[0][2].item()
    wl = win_rate - lose_rate
    
    m_value = m_tensor[0][0].item() if m_tensor.dim() > 1 else m_tensor.item()
    
    policy_dict = policy_tensor_to_move_dict(policy_tensor, fen)
    
    return {
        'q': wl,
        'd': draw_rate,
        'm': m_value,
        'p': policy_dict,
    }


def create_search_backend_eval_fn(model_name: str = DEFAULT_MODEL_NAME) -> Callable[[str], dict[str, object]]:
    """Create an evaluation function for SimpleBackend.

    Args:
        model_name: Model name.

    Returns:
        Eval function that takes a FEN string and returns an evaluation dict.
    """
    def eval_fn(fen: str) -> dict[str, object]:
        return evaluate_position_for_search(fen, model_name)
    return eval_fn


def run_mcts_search(
    fen: str,
    max_playouts: int = 100,
    target_minibatch_size: int = 8,
    cpuct: float = 1.0,
    max_depth: int = 10,
    model_name: str = DEFAULT_MODEL_NAME,
) -> dict[str, Any]:
    """Run MCTS search and return the best move.

    Args:
        fen: FEN string.
        max_playouts: Maximum number of playouts.
        target_minibatch_size: Target minibatch size.
        cpuct: UCT exploration coefficient.
        max_depth: Maximum search depth.
        model_name: Model name.

    Returns:
        Dict with best move and search statistics.
    """
    from .search import SearchParams, Search, SimpleBackend, Node
    
    params = SearchParams(
        max_playouts=max_playouts,
        target_minibatch_size=target_minibatch_size,
        cpuct=cpuct,
        max_depth=max_depth,
    )
    
    eval_fn = create_search_backend_eval_fn(model_name)
    backend = SimpleBackend(model_eval_fn=eval_fn)
    root_node = Node(fen=fen)
    search = Search(
        root_node=root_node,
        backend=backend,
        params=params,
    )
    search.run_blocking()
    best_move = search.get_best_move()
    
    result: dict[str, Any] = {
        'best_move': best_move.uci() if best_move else None,
        'total_playouts': search.get_total_playouts(),
        'max_depth_reached': search.get_current_max_depth(),
        'root_visits': root_node.get_n(),
    }
    
    if root_node.has_children():
        children_stats = []
        for i in range(root_node.get_num_edges()):
            edge = root_node.get_edge_at_index(i)
            child = root_node.get_child_at_index(i)
            if edge:
                move = edge.get_move()
                visits = child.get_n() if child else 0
                q = child.get_q(0.0) if child and child.get_n() > 0 else 0.0
                children_stats.append({
                    'move': move.uci() if move else None,
                    'visits': visits,
                    'q': q,
                    'policy': edge.get_p(),
                })
        children_stats.sort(key=lambda x: x['visits'], reverse=True)
        result['top_moves'] = children_stats[:10]
    
    return result
