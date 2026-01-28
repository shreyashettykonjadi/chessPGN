from typing import List
import chess


def infer_moves_from_fens(
    fen_timeline: List[str],
    *,
    debug: bool = False,
    allow_ambiguous: bool = False
) -> List[chess.Move]:
    """
    Infer chess moves from a deduplicated sequence of FEN positions.

    Behavior:
    - For each consecutive pair (prev_fen, curr_fen):
      - Load prev_fen into python-chess Board
      - Enumerate all legal moves, apply each, and compare board.board_fen()
        against the board portion of curr_fen
      - Decision:
        - Exactly 1 matching move: accept
        - Multiple matches: accept only if allow_ambiguous=True (choose the first)
        - 0 matches: skip silently
    - Never raises; returns a list of chess.Move

    Args:
        fen_timeline: list of FEN strings (deduplicated, stable)
        debug: enable diagnostic logs
        allow_ambiguous: if True, accept transitions with multiple matches (pick first)

    Returns:
        List[chess.Move] inferred between consecutive FENs
    """
    moves: List[chess.Move] = []
    if not fen_timeline or len(fen_timeline) < 2:
        return moves

    for i in range(len(fen_timeline) - 1):
        fen_before = fen_timeline[i]
        fen_after = fen_timeline[i + 1]

        try:
            board_before = chess.Board(fen_before)
            # Use only board portion from FEN after
            board_after_fen = chess.Board(fen_after).board_fen()
        except Exception:
            # Invalid/unsupported FENs: skip this pair
            continue

        legal_moves = list(board_before.legal_moves)
        matches = []

        if debug:
            print(f"[move-infer] pair {i} \u2192 {i+1}")
            print(f"[move-infer] legal moves tried: {len(legal_moves)}")

        for mv in legal_moves:
            try:
                board_before.push(mv)
                if board_before.board_fen() == board_after_fen:
                    matches.append(mv)
            except Exception:
                # Defensive: if push fails, ignore and continue
                pass
            finally:
                # Ensure board is restored
                if board_before.move_stack:
                    board_before.pop()

        if debug:
            print(f"[move-infer] matches found: {len(matches)}")

        if len(matches) == 1:
            moves.append(matches[0])
        elif len(matches) > 1 and allow_ambiguous:
            # Deterministically choose the first match to keep pipeline stable
            moves.append(matches[0])
        # else: 0 matches or ambiguous without permission -> skip silently

    return moves
