import chess
from typing import Any, cast
from src.fen_transition_validator import FENTransitionValidator



def test_simple_pawn_move():
    """
    White pawn moves from e2 to e4
    """

    fen_before = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"
    fen_after  = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b - - 0 1"

    validator = FENTransitionValidator(debug=True)

    is_valid, reason, _info = validator.validate(cast(Any, (fen_before, fen_after)))

    assert is_valid, f"Expected valid pawn move, got rejection: {reason}"


def test_illegal_transition():
    """
    Too many pieces change — must be rejected
    """

    fen_before = "8/8/8/8/8/8/8/8 w - - 0 1"
    fen_after  = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"

    validator = FENTransitionValidator(debug=True)

    is_valid, reason, _info = validator.validate(cast(Any, (fen_before, fen_after)))

    assert not is_valid, "Illegal transition should be rejected"
