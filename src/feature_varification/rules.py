from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

import chess

from chess_utils import get_piece_type_pos, get_start_end_pos_from_move_uci

from .types import RuleEvaluation


def _empty_mask() -> list[bool]:
    return [False] * 64


def _mask_from_positions(positions: list[int]) -> list[bool]:
    mask = _empty_mask()
    for pos in positions:
        if 0 <= pos < 64:
            mask[pos] = True
    return mask


def _flip_square_for_bt4(board: chess.Board, square: chess.Square) -> int:
    """Convert a python-chess square index into the BT4 position index.

    BT4 keeps files fixed but flips ranks when it is black to move. This helper mirrors
    the behaviour already used by ``src.chess_utils.get_pos_from_square``.
    """

    if board.turn == chess.BLACK:
        file_idx = chess.square_file(square)
        rank_idx = chess.square_rank(square)
        square = chess.square(file_idx, 7 - rank_idx)
    return int(square)


def _parse_piece_type(board: chess.Board, piece_type: str) -> tuple[chess.Color, chess.PieceType]:
    """Resolve a project-style piece spec such as ``"own q"`` into chess enums."""

    owner, piece_code = piece_type.split()
    color = board.turn if owner == "own" else (not board.turn)
    piece_map = {
        "p": chess.PAWN,
        "n": chess.KNIGHT,
        "b": chess.BISHOP,
        "r": chess.ROOK,
        "q": chess.QUEEN,
        "k": chess.KING,
    }
    return color, piece_map[piece_code]


def _get_piece_squares(board: chess.Board, piece_type: str) -> list[chess.Square]:
    """Return python-chess squares occupied by the requested piece type."""

    color, piece_type_enum = _parse_piece_type(board, piece_type)
    return list(board.pieces(piece_type_enum, color))


def _collect_ray_squares(
    board: chess.Board,
    square: chess.Square,
    deltas: Sequence[tuple[int, int]],
    *,
    max_steps: int = 7,
    stop_at_blockers: bool = False,
) -> list[int]:
    """Collect squares by walking rays from one origin square.

    Parameters
    ----------
    deltas:
        One or more direction vectors in ``(file_delta, rank_delta)`` format.
    max_steps:
        Maximum number of squares to travel in each direction.
    stop_at_blockers:
        If ``True``, stop extending a ray after the first occupied square.
        If ``False``, geometry ignores occupancy and spans the whole line.
    """

    positions: list[int] = []
    start_file = chess.square_file(square)
    start_rank = chess.square_rank(square)
    for df, dr in deltas:
        for step in range(1, max_steps + 1):
            file_idx = start_file + df * step
            rank_idx = start_rank + dr * step
            if not (0 <= file_idx < 8 and 0 <= rank_idx < 8):
                break
            target_square = chess.square(file_idx, rank_idx)
            positions.append(_flip_square_for_bt4(board, target_square))
            if stop_at_blockers and board.piece_at(target_square) is not None:
                break
    return positions


_RAY_DIRECTION_PRESETS: dict[str, tuple[tuple[int, int], ...]] = {
    "rank": ((1, 0), (-1, 0)),
    "file": ((0, 1), (0, -1)),
    "orthogonal": ((1, 0), (-1, 0), (0, 1), (0, -1)),
    "diagonal": ((1, 1), (1, -1), (-1, 1), (-1, -1)),
    "queen": ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)),
}


class VerificationRule(Protocol):
    """Protocol implemented by all feature verification rules."""

    name: str
    requires_move_uci: bool

    def evaluate(self, fen: str, move_uci: str | None = None) -> RuleEvaluation:
        ...


@dataclass(frozen=True)
class FunctionalRule:
    """Wrap an arbitrary Python callable as a verification rule.

    This is the easiest way to migrate existing one-off notebook logic:
    write a function ``fn(fen, move_uci) -> list[int]`` and register it here.
    """

    name: str
    fn: Callable[[str, str | None], list[int]]
    requires_move_uci: bool = False

    def evaluate(self, fen: str, move_uci: str | None = None) -> RuleEvaluation:
        positions = self.fn(fen, move_uci)
        return RuleEvaluation(mask=_mask_from_positions(positions))


@dataclass(frozen=True)
class AnyOfRule:
    """Union of several rules.

    This is the cleanest replacement for many notebook-style rules that were
    written as ``pos_a + pos_b + pos_c``. A square is positive if *any* child
    rule marks it as positive.
    """

    name: str
    rules: tuple[VerificationRule, ...]
    requires_move_uci: bool = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "requires_move_uci", any(rule.requires_move_uci for rule in self.rules))

    def evaluate(self, fen: str, move_uci: str | None = None) -> RuleEvaluation:
        merged = _empty_mask()
        metadata: dict[str, Any] = {"children": []}
        for rule in self.rules:
            result = rule.evaluate(fen, move_uci)
            merged = [left or right for left, right in zip(merged, result.mask)]
            metadata["children"].append({"name": rule.name, "metadata": result.metadata})
        return RuleEvaluation(mask=merged, metadata=metadata)


@dataclass(frozen=True)
class AllOfRule:
    """Intersection of several rules.

    This is useful when a feature should fire only on squares that satisfy
    multiple conditions simultaneously, for example:

    - queen destinations
    - around opponent king
    - and gives check
    """

    name: str
    rules: tuple[VerificationRule, ...]
    requires_move_uci: bool = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "requires_move_uci", any(rule.requires_move_uci for rule in self.rules))

    def evaluate(self, fen: str, move_uci: str | None = None) -> RuleEvaluation:
        merged: list[bool] | None = None
        metadata: dict[str, Any] = {"children": []}
        for rule in self.rules:
            result = rule.evaluate(fen, move_uci)
            merged = result.mask if merged is None else [left and right for left, right in zip(merged, result.mask)]
            metadata["children"].append({"name": rule.name, "metadata": result.metadata})
        return RuleEvaluation(mask=merged or _empty_mask(), metadata=metadata)


@dataclass(frozen=True)
class PieceTypeRule:
    """Match squares occupied by a specific own/opponent piece type.

    ``piece_type`` follows the existing project convention, e.g. ``"own k"``,
    ``"opponent q"``, ``"own r"``.
    """

    piece_type: str
    name: str | None = None
    requires_move_uci: bool = False

    def __post_init__(self) -> None:
        if self.name is None:
            object.__setattr__(self, "name", f"piece_type::{self.piece_type}")

    def evaluate(self, fen: str, move_uci: str | None = None) -> RuleEvaluation:
        positions = get_piece_type_pos(fen, self.piece_type)
        return RuleEvaluation(mask=_mask_from_positions(positions), metadata={"piece_type": self.piece_type})


@dataclass(frozen=True)
class EmptySquareRule:
    """Match empty board squares."""

    name: str = "empty_squares"
    requires_move_uci: bool = False

    def evaluate(self, fen: str, move_uci: str | None = None) -> RuleEvaluation:
        board = chess.Board(fen)
        positions = [_flip_square_for_bt4(board, square) for square in chess.SQUARES if board.piece_at(square) is None]
        return RuleEvaluation(mask=_mask_from_positions(positions))


@dataclass(frozen=True)
class OccupiedSquareRule:
    """Match all occupied squares, optionally restricted to one side."""

    owner: str | None = None
    name: str | None = None
    requires_move_uci: bool = False

    def __post_init__(self) -> None:
        if self.owner not in {None, "own", "opponent"}:
            raise ValueError(f"Unsupported owner: {self.owner}")
        if self.name is None:
            object.__setattr__(self, "name", f"occupied::{self.owner or 'any'}")

    def evaluate(self, fen: str, move_uci: str | None = None) -> RuleEvaluation:
        board = chess.Board(fen)
        color = None if self.owner is None else (board.turn if self.owner == "own" else not board.turn)
        positions = [
            _flip_square_for_bt4(board, square)
            for square, piece in board.piece_map().items()
            if color is None or piece.color == color
        ]
        return RuleEvaluation(mask=_mask_from_positions(positions), metadata={"owner": self.owner})


@dataclass(frozen=True)
class PieceNeighborhoodRule:
    """Match squares in a local neighbourhood around one or more pieces.

    This generalizes rules such as:
    - around own king
    - surrounding opponent rook
    - squares adjacent to a knight

    Parameters
    ----------
    piece_type:
        Project-style piece selector, e.g. ``"own k"`` or ``"opponent r"``.
    radius:
        Chebyshev radius around each matching piece.
    include_center:
        Whether the piece's own square should also be marked positive.
    """

    piece_type: str
    radius: int = 1
    include_center: bool = True
    name: str | None = None
    requires_move_uci: bool = False

    def __post_init__(self) -> None:
        if self.name is None:
            object.__setattr__(self, "name", f"neighborhood::{self.piece_type}::r{self.radius}")

    def evaluate(self, fen: str, move_uci: str | None = None) -> RuleEvaluation:
        board = chess.Board(fen)
        positions: set[int] = set()
        for square in _get_piece_squares(board, self.piece_type):
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            for dr in range(-self.radius, self.radius + 1):
                for df in range(-self.radius, self.radius + 1):
                    if not self.include_center and dr == 0 and df == 0:
                        continue
                    nr = rank + dr
                    nf = file + df
                    if 0 <= nr < 8 and 0 <= nf < 8:
                        positions.add(_flip_square_for_bt4(board, chess.square(nf, nr)))
        return RuleEvaluation(mask=_mask_from_positions(sorted(positions)), metadata={"piece_type": self.piece_type})


@dataclass(frozen=True)
class PieceRayRule:
    """Match squares on geometric rays emitted from one or more pieces.

    This class is the reusable replacement for many notebook rules that talk
    about "same rank", "same file", or "same diagonal" relative to a piece.

    Examples
    --------
    ``PieceRayRule("own r", directions=("rank", "file"))``
        Squares on the rook-like lines of own rooks.

    ``PieceRayRule("opponent b", directions=("diagonal",), stop_at_blockers=True)``
        Squares on bishop diagonals, stopping once a blocker is reached.
    """

    piece_type: str
    directions: tuple[str, ...]
    max_steps: int = 7
    stop_at_blockers: bool = False
    include_origin: bool = False
    name: str | None = None
    requires_move_uci: bool = False

    def __post_init__(self) -> None:
        invalid = [direction for direction in self.directions if direction not in _RAY_DIRECTION_PRESETS]
        if invalid:
            raise ValueError(f"Unsupported ray directions: {invalid}")
        if self.name is None:
            joined = "+".join(self.directions)
            object.__setattr__(self, "name", f"ray::{self.piece_type}::{joined}")

    def evaluate(self, fen: str, move_uci: str | None = None) -> RuleEvaluation:
        board = chess.Board(fen)
        positions: set[int] = set()
        deltas = tuple(delta for direction in self.directions for delta in _RAY_DIRECTION_PRESETS[direction])
        for square in _get_piece_squares(board, self.piece_type):
            if self.include_origin:
                positions.add(_flip_square_for_bt4(board, square))
            positions.update(
                _collect_ray_squares(
                    board,
                    square,
                    deltas,
                    max_steps=self.max_steps,
                    stop_at_blockers=self.stop_at_blockers,
                )
            )
        return RuleEvaluation(
            mask=_mask_from_positions(sorted(positions)),
            metadata={
                "piece_type": self.piece_type,
                "directions": list(self.directions),
                "max_steps": self.max_steps,
                "stop_at_blockers": self.stop_at_blockers,
            },
        )


@dataclass(frozen=True)
class PieceFrontSpanRule:
    """Match squares in front of one or more pieces.

    "In front of" is interpreted in the chess sense: towards promotion for the
    owner of the piece. This makes the rule color-aware without introducing
    separate white/black logic in notebooks.

    Parameters
    ----------
    piece_type:
        Piece selector such as ``"own p"`` or ``"opponent k"``.
    max_steps:
        How far the forward span should extend.
    include_same_file:
        Whether to include the forward ray on the piece's own file.
    include_adjacent_files:
        Whether to also include forward rays on the neighboring files. This is
        useful for "front cone" style rules.
    stop_at_blockers:
        Whether to stop each ray once an occupied square is reached.
    """

    piece_type: str
    max_steps: int = 7
    include_same_file: bool = True
    include_adjacent_files: bool = False
    stop_at_blockers: bool = False
    include_origin: bool = False
    name: str | None = None
    requires_move_uci: bool = False

    def __post_init__(self) -> None:
        if not self.include_same_file and not self.include_adjacent_files:
            raise ValueError("At least one of include_same_file / include_adjacent_files must be True")
        if self.name is None:
            shape = "cone" if self.include_adjacent_files else "file"
            object.__setattr__(self, "name", f"front::{self.piece_type}::{shape}")

    def evaluate(self, fen: str, move_uci: str | None = None) -> RuleEvaluation:
        board = chess.Board(fen)
        positions: set[int] = set()
        color, _ = _parse_piece_type(board, self.piece_type)
        forward_rank_delta = 1 if color == chess.WHITE else -1

        file_offsets = [0] if self.include_same_file else []
        if self.include_adjacent_files:
            file_offsets.extend([-1, 1])

        for square in _get_piece_squares(board, self.piece_type):
            if self.include_origin:
                positions.add(_flip_square_for_bt4(board, square))
            start_file = chess.square_file(square)
            start_rank = chess.square_rank(square)
            for file_offset in file_offsets:
                for step in range(1, self.max_steps + 1):
                    file_idx = start_file + file_offset
                    rank_idx = start_rank + forward_rank_delta * step
                    if not (0 <= file_idx < 8 and 0 <= rank_idx < 8):
                        break
                    target_square = chess.square(file_idx, rank_idx)
                    positions.add(_flip_square_for_bt4(board, target_square))
                    if self.stop_at_blockers and board.piece_at(target_square) is not None:
                        break

        return RuleEvaluation(
            mask=_mask_from_positions(sorted(positions)),
            metadata={
                "piece_type": self.piece_type,
                "max_steps": self.max_steps,
                "include_same_file": self.include_same_file,
                "include_adjacent_files": self.include_adjacent_files,
                "stop_at_blockers": self.stop_at_blockers,
            },
        )


@dataclass(frozen=True)
class RelativeOffsetRule:
    """Match squares at fixed offsets from one or more pieces.

    This is the most flexible geometric rule. It can express patterns like:

    - one square in front of a pawn
    - knight jump targets around a king
    - custom local motifs discovered during manual inspection

    Offsets are given in board coordinates as ``(file_delta, rank_delta)``.
    If ``orient_to_owner`` is ``True``, positive rank means "forward for the
    piece owner" instead of "towards White's side of the board".
    """

    piece_type: str
    offsets: tuple[tuple[int, int], ...]
    orient_to_owner: bool = False
    include_origin: bool = False
    name: str | None = None
    requires_move_uci: bool = False

    def __post_init__(self) -> None:
        if self.name is None:
            object.__setattr__(self, "name", f"offset::{self.piece_type}")

    def evaluate(self, fen: str, move_uci: str | None = None) -> RuleEvaluation:
        board = chess.Board(fen)
        positions: set[int] = set()
        color, _ = _parse_piece_type(board, self.piece_type)
        orientation = 1 if (not self.orient_to_owner or color == chess.WHITE) else -1

        for square in _get_piece_squares(board, self.piece_type):
            start_file = chess.square_file(square)
            start_rank = chess.square_rank(square)
            if self.include_origin:
                positions.add(_flip_square_for_bt4(board, square))
            for df, dr in self.offsets:
                file_idx = start_file + df
                rank_idx = start_rank + dr * orientation
                if 0 <= file_idx < 8 and 0 <= rank_idx < 8:
                    positions.add(_flip_square_for_bt4(board, chess.square(file_idx, rank_idx)))

        return RuleEvaluation(
            mask=_mask_from_positions(sorted(positions)),
            metadata={"piece_type": self.piece_type, "offsets": [list(offset) for offset in self.offsets]},
        )


@dataclass(frozen=True)
class MoveStartSquareRule:
    """Match the source square of a move."""

    name: str = "move_start_square"
    requires_move_uci: bool = True

    def evaluate(self, fen: str, move_uci: str | None = None) -> RuleEvaluation:
        if not move_uci:
            raise ValueError("MoveStartSquareRule requires move_uci")
        start_pos, _ = get_start_end_pos_from_move_uci(fen, move_uci)
        return RuleEvaluation(mask=_mask_from_positions([start_pos]), metadata={"move_uci": move_uci})


@dataclass(frozen=True)
class MoveEndSquareRule:
    """Match the destination square of a move."""

    name: str = "move_end_square"
    requires_move_uci: bool = True

    def evaluate(self, fen: str, move_uci: str | None = None) -> RuleEvaluation:
        if not move_uci:
            raise ValueError("MoveEndSquareRule requires move_uci")
        _, end_pos = get_start_end_pos_from_move_uci(fen, move_uci)
        return RuleEvaluation(mask=_mask_from_positions([end_pos]), metadata={"move_uci": move_uci})


@dataclass(frozen=True)
class MoveStartPieceRule:
    """Match a supervised move source only when it contains ``piece_type``."""

    piece_type: str
    name: str | None = None
    requires_move_uci: bool = True

    def __post_init__(self) -> None:
        if self.name is None:
            object.__setattr__(self, "name", f"move_start::{self.piece_type}")

    def evaluate(self, fen: str, move_uci: str | None = None) -> RuleEvaluation:
        if not move_uci:
            raise ValueError("MoveStartPieceRule requires move_uci")
        board = chess.Board(fen)
        color, piece_enum = _parse_piece_type(board, self.piece_type)
        move = chess.Move.from_uci(move_uci)
        piece = board.piece_at(move.from_square)
        matches = piece is not None and piece.color == color and piece.piece_type == piece_enum
        positions = [_flip_square_for_bt4(board, move.from_square)] if matches else []
        return RuleEvaluation(
            mask=_mask_from_positions(positions),
            metadata={"move_uci": move_uci, "piece_type": self.piece_type, "matches": matches},
        )


@dataclass(frozen=True)
class MoveEndPieceRule:
    """Match a supervised move destination for a selected moving piece.

    ``target_owner`` optionally restricts whether the destination is empty,
    occupied by an own piece, or occupied by an opponent piece before the move.
    """

    piece_type: str
    target_owner: str | None = None
    name: str | None = None
    requires_move_uci: bool = True

    def __post_init__(self) -> None:
        if self.target_owner not in {None, "empty", "own", "opponent"}:
            raise ValueError(f"Unsupported target owner: {self.target_owner}")
        if self.name is None:
            object.__setattr__(self, "name", f"move_end::{self.piece_type}::{self.target_owner or 'any'}")

    def evaluate(self, fen: str, move_uci: str | None = None) -> RuleEvaluation:
        if not move_uci:
            raise ValueError("MoveEndPieceRule requires move_uci")
        board = chess.Board(fen)
        color, piece_enum = _parse_piece_type(board, self.piece_type)
        move = chess.Move.from_uci(move_uci)
        piece = board.piece_at(move.from_square)
        target = board.piece_at(move.to_square)
        matches_piece = piece is not None and piece.color == color and piece.piece_type == piece_enum
        matches_target = (
            self.target_owner is None
            or (self.target_owner == "empty" and target is None)
            or (self.target_owner == "own" and target is not None and target.color == board.turn)
            or (self.target_owner == "opponent" and target is not None and target.color != board.turn)
        )
        matches = matches_piece and matches_target
        positions = [_flip_square_for_bt4(board, move.to_square)] if matches else []
        return RuleEvaluation(
            mask=_mask_from_positions(positions),
            metadata={
                "move_uci": move_uci,
                "piece_type": self.piece_type,
                "target_owner": self.target_owner,
                "matches": matches,
            },
        )


@dataclass(frozen=True)
class KingNeighborhoodRule:
    """Match the 3x3 neighbourhood around either own king or opponent king."""

    owner: str = "own"
    name: str | None = None
    requires_move_uci: bool = False

    def __post_init__(self) -> None:
        if self.owner not in {"own", "opponent"}:
            raise ValueError(f"Unsupported owner: {self.owner}")
        if self.name is None:
            object.__setattr__(self, "name", f"{self.owner}_king_neighborhood")

    def evaluate(self, fen: str, move_uci: str | None = None) -> RuleEvaluation:
        board = chess.Board(fen)
        color = board.turn if self.owner == "own" else (not board.turn)
        king_square = board.king(color)
        if king_square is None:
            return RuleEvaluation(mask=_empty_mask(), metadata={"owner": self.owner})

        positions: list[int] = []
        rank = chess.square_rank(king_square)
        file = chess.square_file(king_square)
        for dr in (-1, 0, 1):
            for df in (-1, 0, 1):
                nr = rank + dr
                nf = file + df
                if 0 <= nr < 8 and 0 <= nf < 8:
                    positions.append(_flip_square_for_bt4(board, chess.square(nf, nr)))
        return RuleEvaluation(mask=_mask_from_positions(positions), metadata={"owner": self.owner})


@dataclass(frozen=True)
class PieceDestinationRule:
    """Match legal destination squares of a specific own/opponent piece type.

    This covers many movement-style taxonomy notebooks. Example:
    ``PieceDestinationRule(piece_type="own n")`` matches all legal knight destinations
    for the side to move.
    """

    piece_type: str
    name: str | None = None
    requires_move_uci: bool = False

    def __post_init__(self) -> None:
        if self.name is None:
            object.__setattr__(self, "name", f"destinations::{self.piece_type}")

    def evaluate(self, fen: str, move_uci: str | None = None) -> RuleEvaluation:
        board = chess.Board(fen)
        color, target_piece_type = _parse_piece_type(board, self.piece_type)

        # ``board.legal_moves`` only enumerates the current side to move. For
        # opponent-piece movement motifs we need to temporarily view the same
        # position from the opponent's turn instead of silently returning an
        # empty set.
        move_board = board if color == board.turn else board.copy(stack=False)
        move_board.turn = color

        positions: list[int] = []
        for move in move_board.legal_moves:
            piece = move_board.piece_at(move.from_square)
            if piece is None:
                continue
            if piece.color != color or piece.piece_type != target_piece_type:
                continue
            positions.append(_flip_square_for_bt4(board, move.to_square))

        return RuleEvaluation(mask=_mask_from_positions(sorted(set(positions))), metadata={"piece_type": self.piece_type})


@dataclass(frozen=True)
class PieceAttackRule:
    """Match squares attacked by a specific own/opponent piece type.

    Unlike ``PieceDestinationRule``, this includes defended friendly pieces and
    pseudo-legal capture squares. It is therefore suitable for attack and
    protection interpretations rather than movement-only interpretations.
    """

    piece_type: str
    name: str | None = None
    requires_move_uci: bool = False

    def __post_init__(self) -> None:
        if self.name is None:
            object.__setattr__(self, "name", f"attacks::{self.piece_type}")

    def evaluate(self, fen: str, move_uci: str | None = None) -> RuleEvaluation:
        board = chess.Board(fen)
        positions: set[int] = set()
        for square in _get_piece_squares(board, self.piece_type):
            positions.update(_flip_square_for_bt4(board, target) for target in board.attacks(square))
        return RuleEvaluation(
            mask=_mask_from_positions(sorted(positions)),
            metadata={"piece_type": self.piece_type},
        )


@dataclass(frozen=True)
class ProtectedPieceRule:
    """Match pieces of ``piece_type`` defended by the requested side."""

    piece_type: str
    protector_owner: str | None = None
    name: str | None = None
    requires_move_uci: bool = False

    def __post_init__(self) -> None:
        if self.protector_owner not in {None, "own", "opponent"}:
            raise ValueError(f"Unsupported protector owner: {self.protector_owner}")
        if self.name is None:
            protector = self.protector_owner or self.piece_type.split()[0]
            object.__setattr__(self, "name", f"protected::{self.piece_type}::by_{protector}")

    def evaluate(self, fen: str, move_uci: str | None = None) -> RuleEvaluation:
        board = chess.Board(fen)
        default_owner, _ = self.piece_type.split()
        protector_owner = self.protector_owner or default_owner
        protector_color = board.turn if protector_owner == "own" else not board.turn
        positions = [
            _flip_square_for_bt4(board, square)
            for square in _get_piece_squares(board, self.piece_type)
            if board.attackers(protector_color, square)
        ]
        return RuleEvaluation(
            mask=_mask_from_positions(positions),
            metadata={"piece_type": self.piece_type, "protector_owner": protector_owner},
        )


@dataclass(frozen=True)
class CapturablePieceRule:
    """Match opponent pieces that a selected side can capture immediately."""

    attacker_piece_type: str | None = None
    attacker_owner: str = "own"
    target_piece_type: str | None = None
    name: str | None = None
    requires_move_uci: bool = False

    def __post_init__(self) -> None:
        if self.attacker_owner not in {"own", "opponent"}:
            raise ValueError(f"Unsupported attacker owner: {self.attacker_owner}")
        if self.attacker_piece_type and not self.attacker_piece_type.startswith(f"{self.attacker_owner} "):
            raise ValueError("attacker_piece_type owner must match attacker_owner")
        if self.name is None:
            attacker = self.attacker_piece_type or f"{self.attacker_owner} any"
            target = self.target_piece_type or "opposing any"
            object.__setattr__(self, "name", f"capturable::{target}::by_{attacker}")

    def evaluate(self, fen: str, move_uci: str | None = None) -> RuleEvaluation:
        board = chess.Board(fen)
        attacker_color = board.turn if self.attacker_owner == "own" else not board.turn
        target_color = not attacker_color
        attacker_enum = _parse_piece_type(board, self.attacker_piece_type)[1] if self.attacker_piece_type else None
        target_enum = _parse_piece_type(board, self.target_piece_type)[1] if self.target_piece_type else None
        positions: set[int] = set()
        for square, piece in board.piece_map().items():
            if piece.color != target_color or (target_enum is not None and piece.piece_type != target_enum):
                continue
            attackers = board.attackers(attacker_color, square)
            if attacker_enum is not None:
                attackers = chess.SquareSet(
                    source for source in attackers if board.piece_type_at(source) == attacker_enum
                )
            if attackers:
                positions.add(_flip_square_for_bt4(board, square))
        return RuleEvaluation(
            mask=_mask_from_positions(sorted(positions)),
            metadata={
                "attacker_piece_type": self.attacker_piece_type,
                "attacker_owner": self.attacker_owner,
                "target_piece_type": self.target_piece_type,
            },
        )


@dataclass(frozen=True)
class CheckingMoveDestinationRule:
    """Match legal move destinations that give check."""

    piece_type: str | None = None
    name: str | None = None
    requires_move_uci: bool = False

    def __post_init__(self) -> None:
        if self.piece_type and not self.piece_type.startswith("own "):
            raise ValueError("Checking moves must use an own piece selector")
        if self.name is None:
            object.__setattr__(self, "name", f"checking_destinations::{self.piece_type or 'own any'}")

    def evaluate(self, fen: str, move_uci: str | None = None) -> RuleEvaluation:
        board = chess.Board(fen)
        piece_enum = _parse_piece_type(board, self.piece_type)[1] if self.piece_type else None
        positions: set[int] = set()
        for move in board.legal_moves:
            piece = board.piece_at(move.from_square)
            if piece is None or (piece_enum is not None and piece.piece_type != piece_enum):
                continue
            if board.gives_check(move):
                positions.add(_flip_square_for_bt4(board, move.to_square))
        return RuleEvaluation(
            mask=_mask_from_positions(sorted(positions)),
            metadata={"piece_type": self.piece_type},
        )


@dataclass(frozen=True)
class RelativeBoardRegionRule:
    """Match a named region in side-to-move coordinates."""

    region: str
    name: str | None = None
    requires_move_uci: bool = False

    def __post_init__(self) -> None:
        supported = {
            "own_back_rank",
            "own_two_ranks",
            "opponent_back_rank",
            "opponent_two_ranks",
            "center_four",
            "extended_center",
            "edge",
            "queenside",
            "kingside",
        }
        if self.region not in supported:
            raise ValueError(f"Unsupported board region: {self.region}")
        if self.name is None:
            object.__setattr__(self, "name", f"region::{self.region}")

    def evaluate(self, fen: str, move_uci: str | None = None) -> RuleEvaluation:
        board = chess.Board(fen)
        positions: list[int] = []
        for square in chess.SQUARES:
            file_idx = chess.square_file(square)
            rank_idx = chess.square_rank(square)
            own_rank = rank_idx if board.turn == chess.WHITE else 7 - rank_idx
            matches = {
                "own_back_rank": own_rank == 0,
                "own_two_ranks": own_rank <= 1,
                "opponent_back_rank": own_rank == 7,
                "opponent_two_ranks": own_rank >= 6,
                "center_four": file_idx in {3, 4} and rank_idx in {3, 4},
                "extended_center": file_idx in {2, 3, 4, 5} and rank_idx in {2, 3, 4, 5},
                "edge": file_idx in {0, 7} or rank_idx in {0, 7},
                "queenside": file_idx <= 3,
                "kingside": file_idx >= 4,
            }[self.region]
            if matches:
                positions.append(_flip_square_for_bt4(board, square))
        return RuleEvaluation(mask=_mask_from_positions(positions), metadata={"region": self.region})


@dataclass(frozen=True)
class QueenCheckAroundOpponentKingRule:
    """Match squares where the moving side's queen can move and give immediate check.

    This directly captures one recurring style of rule from the taxonomy notebooks:
    "queen moves to a square around the opponent king and checks".
    """

    name: str = "queen_check_around_opponent_king"
    requires_move_uci: bool = False

    def evaluate(self, fen: str, move_uci: str | None = None) -> RuleEvaluation:
        board = chess.Board(fen)
        opp_king = board.king(not board.turn)
        if opp_king is None:
            return RuleEvaluation(mask=_empty_mask())

        king_neighbors: set[chess.Square] = set()
        rank = chess.square_rank(opp_king)
        file = chess.square_file(opp_king)
        for dr in (-1, 0, 1):
            for df in (-1, 0, 1):
                nr = rank + dr
                nf = file + df
                if 0 <= nr < 8 and 0 <= nf < 8:
                    king_neighbors.add(chess.square(nf, nr))

        positions: list[int] = []
        for move in board.legal_moves:
            piece = board.piece_at(move.from_square)
            if piece is None or piece.color != board.turn or piece.piece_type != chess.QUEEN:
                continue
            if move.to_square not in king_neighbors:
                continue
            if board.gives_check(move):
                positions.append(_flip_square_for_bt4(board, move.to_square))

        return RuleEvaluation(mask=_mask_from_positions(sorted(set(positions))))


RULE_REGISTRY: dict[str, VerificationRule] = {
    "move_start_square": MoveStartSquareRule(),
    "move_end_square": MoveEndSquareRule(),
    "own_king_neighborhood": KingNeighborhoodRule(owner="own"),
    "opponent_king_neighborhood": KingNeighborhoodRule(owner="opponent"),
    "queen_check_around_opponent_king": QueenCheckAroundOpponentKingRule(),
}


def register_rule(rule: VerificationRule, *, overwrite: bool = False) -> None:
    """Register a reusable rule under ``rule.name``.

    Notebooks often define one-off helper functions inline. Once a rule has
    stabilized, registering it here makes it importable everywhere else and
    keeps threshold / evaluation logic separate from rule definition.
    """

    if not overwrite and rule.name in RULE_REGISTRY:
        raise ValueError(f"Rule '{rule.name}' is already registered")
    RULE_REGISTRY[rule.name] = rule


def same_rank_rule(piece_type: str, **kwargs: Any) -> PieceRayRule:
    """Convenience wrapper for "same horizontal line as this piece"."""

    return PieceRayRule(piece_type=piece_type, directions=("rank",), **kwargs)


def same_file_rule(piece_type: str, **kwargs: Any) -> PieceRayRule:
    """Convenience wrapper for "same vertical line as this piece"."""

    return PieceRayRule(piece_type=piece_type, directions=("file",), **kwargs)


def same_diagonal_rule(piece_type: str, **kwargs: Any) -> PieceRayRule:
    """Convenience wrapper for "same diagonal as this piece"."""

    return PieceRayRule(piece_type=piece_type, directions=("diagonal",), **kwargs)


def front_file_rule(piece_type: str, **kwargs: Any) -> PieceFrontSpanRule:
    """Convenience wrapper for squares straight ahead of a piece."""

    return PieceFrontSpanRule(piece_type=piece_type, include_same_file=True, include_adjacent_files=False, **kwargs)


def front_cone_rule(piece_type: str, **kwargs: Any) -> PieceFrontSpanRule:
    """Convenience wrapper for a forward cone: same file plus adjacent files."""

    return PieceFrontSpanRule(piece_type=piece_type, include_same_file=True, include_adjacent_files=True, **kwargs)
