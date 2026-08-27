import chess
import torch
from transformer_lens import HookedTransformer
from lm_saes.lc0_mapping.lc0_mapping import idx_to_uci_mappings
from lm_saes.circuit.leela_board import LeelaBoard


def get_move_from_model(model: HookedTransformer, fen: str, return_list: bool = False) -> str:

    leela_board = LeelaBoard.from_fen(fen, history_synthesis=True)

    output, _  = model.run_with_cache(fen, prepend_bos=False)
    policy_output = output[0][0]

    chess_board = chess.Board(fen)
    legal_uci_set = set(move.uci() for move in chess_board.legal_moves)
    sorted_indices = torch.argsort(policy_output, descending=True)

    top_legal_moves = []
    for idx in sorted_indices:
        uci = leela_board.idx2uci(idx.item())
        if uci in legal_uci_set:
            top_legal_moves.append((uci, policy_output[idx].item()))
        if len(top_legal_moves) >= 5:
            break
    
    if not return_list:
        return top_legal_moves[0][0]
    else:
        return top_legal_moves

def get_wdl_from_model(model: HookedTransformer, fen: str) -> tuple[float, float, float]:
    output, _  = model.run_with_cache(fen, prepend_bos=False)
    value_output = output[1][0]
    win_prob, draw_prob, loss_prob = value_output
    if isinstance(win_prob, torch.Tensor):
        win_prob = win_prob.detach().cpu().item()
        draw_prob = draw_prob.detach().cpu().item()
        loss_prob = loss_prob.detach().cpu().item()
    return win_prob, draw_prob, loss_prob

def get_m_from_model(model: HookedTransformer, fen: str) -> float:
    output, _  = model.run_with_cache(fen, prepend_bos=False)
    m_output = output[2][0][0]
    return m_output

def get_value_from_model(model: HookedTransformer, fen: str) -> float:
    w,_,l = get_wdl_from_model(model, fen)
    return w - l

def get_move_from_policy_output(policy_output: torch.Tensor, fen: str, return_list: bool = False) -> str:
    leela_board = LeelaBoard.from_fen(fen, history_synthesis=True)
    if policy_output.ndim > 1:
        policy_output = policy_output[0]
    chess_board = chess.Board(fen)
    legal_uci_set = set(move.uci() for move in chess_board.legal_moves)
    sorted_indices = torch.argsort(policy_output, descending=True)

    top_legal_moves = []
    for idx in sorted_indices:
        uci = leela_board.idx2uci(idx.item())
        if uci in legal_uci_set:
            top_legal_moves.append((uci, policy_output[idx].item()))
        if len(top_legal_moves) >= 5:
            break
    
    if not return_list:
        return top_legal_moves[0][0]
    else:
        return top_legal_moves

def get_move_from_policy_output_with_logit(
    policy_output: torch.Tensor, fen: str, return_list: bool = False
) -> tuple[str, float] | list[tuple[str, float]]:
    """Get move and its logit value from policy output.

    Parameters
    ----------
    policy_output:
        Policy output tensor with shape [vocab_size] or [batch_size, vocab_size].
    fen:
        FEN string.
    return_list:
        If True, return a list of top 5 legal moves; otherwise return the best move and its logit.

    Returns
    -------
    tuple[str, float] | list[tuple[str, float]]
        If return_list=False, returns (uci, logit) tuple.
        If return_list=True, returns [(uci, logit), ...] list.
    """
    leela_board = LeelaBoard.from_fen(fen, history_synthesis=True)
    if policy_output.ndim > 1:
        policy_output = policy_output[0]
    chess_board = chess.Board(fen)
    legal_uci_set = set(move.uci() for move in chess_board.legal_moves)
    sorted_indices = torch.argsort(policy_output, descending=True)

    top_legal_moves: list[tuple[str, float]] = []
    for idx in sorted_indices:
        uci = leela_board.idx2uci(idx.item())
        if uci in legal_uci_set:
            top_legal_moves.append((uci, float(policy_output[idx].item())))
        if len(top_legal_moves) >= 5:
            break

    if not return_list:
        return top_legal_moves[0]  # Return (uci, logit) tuple
    else:
        return top_legal_moves


def get_move_from_policy_output_with_prob(
    policy_output: torch.Tensor,
    fen: str,
    return_list: bool = False,
    move_uci: str | None = None
) -> tuple[str, float] | list[tuple[str, float, float]] | float | None:
    if policy_output.ndim > 1:
        policy_output = policy_output[0]

    leela_board = LeelaBoard.from_fen(fen, history_synthesis=True)
    chess_board = chess.Board(fen)

    legal_uci_list = [move.uci() for move in chess_board.legal_moves]
    if not legal_uci_list:
        return None if move_uci else ([] if return_list else (None, 0.0))

    valid_moves = []  # [(uci, logit)]
    for uci in legal_uci_list:
        try:
            idx = leela_board.uci2idx(uci)
            logit = policy_output[idx].item()
            valid_moves.append((uci, logit))
        except (KeyError, IndexError):
            # If not in vocabulary, try simplified format (for promotion moves)
            if len(uci) == 5 and uci[4] in ['q', 'r', 'b', 'n']:
                # Try removing promotion suffix
                uci_simple = uci[:4]
                try:
                    idx = leela_board.uci2idx(uci_simple)
                    logit = policy_output[idx].item()
                    # Use original UCI format but with simplified format's logit
                    valid_moves.append((uci, logit))
                except (KeyError, IndexError):
                    # If simplified format is also not in vocabulary, skip
                    continue
            else:
                # Non-promotion move but not in vocabulary, skip
                continue
    
    if not valid_moves:
        # If no valid moves, return None or empty list
        return None if move_uci else ([] if return_list else (None, 0.0))
    
    # Extract valid UCI and logits
    valid_uci_list = [uci for uci, _ in valid_moves]
    legal_logits = torch.tensor([logit for _, logit in valid_moves])
    
    # Calculate probabilities
    legal_probs = torch.softmax(
        legal_logits - legal_logits.max(), dim=0
    )
    
    if move_uci is not None:
        if move_uci not in valid_uci_list:
            return None
        idx = valid_uci_list.index(move_uci)
        return float(legal_probs[idx])
    
    sorted_indices = torch.argsort(legal_logits, descending=True)

    if return_list:
        out: list[tuple[str, float, float]] = []
        for idx in sorted_indices.tolist():
            out.append(
                (
                    valid_uci_list[idx],
                    float(legal_logits[idx].item()),
                    float(legal_probs[idx].item()),
                )
            )
        return out

    best_idx = int(sorted_indices[0].item())
    return valid_uci_list[best_idx], float(legal_probs[best_idx].item())

def get_move_prob_and_rank(model: HookedTransformer, fen: str, move_uci: str):
    output, _ = model.run_with_cache(fen, prepend_bos=False)
    policy_output = output[0]  # [1, vocab_size]
    probs = get_move_from_policy_output_with_prob(policy_output, fen, move_uci=move_uci)
    if probs is None:
        return None
    prob = probs
    legal_moves_with_probs = get_move_from_policy_output_with_prob(policy_output, fen, return_list=True)
    sorted_probs = sorted([p for _, _, p in legal_moves_with_probs], reverse=True)
    rank = sorted_probs.index(prob) + 1

    return prob, rank


def get_move_prob_and_rank_from_policy_output(policy_output: torch.Tensor, fen: str, move_uci: str):
    policy_output = policy_output[0]  # [vocab_size]

    leela_board = LeelaBoard.from_fen(fen, history_synthesis=True)
    chess_board = chess.Board(fen)
    legal_uci_list = [move.uci() for move in chess_board.legal_moves]
    if move_uci not in legal_uci_list:
        return None
    legal_logits = torch.stack([policy_output[leela_board.uci2idx(uci)] for uci in legal_uci_list])
    legal_probs = torch.softmax(legal_logits - legal_logits.max(), dim=0)
    idx = legal_uci_list.index(move_uci)
    prob = float(legal_probs[idx].item())

    sorted_probs, _ = torch.sort(legal_probs, descending=True)
    rank = (sorted_probs >= prob).sum().item()

    return prob, rank

def get_value_from_output(output: list):
    return output[1][0][0] - output[1][0][2]


def get_top_3_moves_and_probs(model: HookedTransformer, fen: str) -> list[tuple[str, float]]:    
    leela_board = LeelaBoard.from_fen(fen, history_synthesis=True)
    output, _  = model.run_with_cache(fen, prepend_bos=False)
    policy_output = output[0]
    
    chess_board = chess.Board(fen)
    legal_uci_set = set(move.uci() for move in chess_board.legal_moves)
    
    sorted_indices = torch.argsort(policy_output[0], descending=True)
    top_moves = []
    for idx in sorted_indices:
        uci = leela_board.idx2uci(idx.item())
        if uci in legal_uci_set:
            prob, rank = get_move_prob_and_rank_from_policy_output(policy_output, fen, uci)
            top_moves.append((uci, prob))
            if len(top_moves) >= 3:
                break
    return top_moves
