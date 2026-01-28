import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fen_timeline import FENTimeline


def test_deduplicates_consecutive_fens():
    """Consecutive duplicate FENs should not be added."""
    # Disable validation for unit tests focused on deduplication
    timeline = FENTimeline(validate_transitions=False)
    
    fen1 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    fen2 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    
    # First FEN should be added
    assert timeline.collect(fen1) == fen1
    assert len(timeline) == 1
    
    # Duplicate should return None
    assert timeline.collect(fen1) is None
    assert len(timeline) == 1
    
    # New FEN should be added
    assert timeline.collect(fen2) == fen2
    assert len(timeline) == 2
    
    # Duplicate of second should return None
    assert timeline.collect(fen2) is None
    assert len(timeline) == 2


def test_handles_none_input():
    """None input should be ignored."""
    timeline = FENTimeline(validate_transitions=False)
    
    assert timeline.collect(None) is None
    assert len(timeline) == 0


def test_history_returns_copy():
    """History should return a copy, not the internal list."""
    timeline = FENTimeline(validate_transitions=False)
    fen1 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    
    timeline.collect(fen1)
    history = timeline.history
    
    # Modifying returned list shouldn't affect internal state
    history.append("fake_fen")
    assert len(timeline) == 1
    assert len(timeline.history) == 1


def test_reset_clears_state():
    """Reset should clear history and last_fen."""
    timeline = FENTimeline(validate_transitions=False)
    fen1 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    
    timeline.collect(fen1)
    assert len(timeline) == 1
    
    timeline.reset()
    assert len(timeline) == 0
    assert timeline.last_fen is None
    
    # Same FEN should now be accepted again
    assert timeline.collect(fen1) == fen1


def test_non_consecutive_duplicates_allowed():
    """Non-consecutive duplicates should be allowed (A, B, A is valid)."""
    timeline = FENTimeline(validate_transitions=False)
    
    fen_a = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    fen_b = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    
    assert timeline.collect(fen_a) == fen_a
    assert timeline.collect(fen_b) == fen_b
    assert timeline.collect(fen_a) == fen_a
    
    assert len(timeline) == 3
    assert timeline.history == [fen_a, fen_b, fen_a]


def test_transition_validation_rejects_invalid_kings():
    """Validation should reject FENs without proper kings."""
    timeline = FENTimeline(validate_transitions=True)
    
    # Valid starting position
    valid_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    assert timeline.collect(valid_fen) == valid_fen
    
    # Invalid: no kings
    no_kings = "8/8/8/8/8/8/8/8 w - - 0 1"
    assert timeline.collect(no_kings) is None
    
    # Invalid: two white kings
    two_white_kings = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBKKBNR w KQkq - 0 1"
    assert timeline.collect(two_white_kings) is None


def test_transition_validation_rejects_too_many_changes():
    """Validation should reject transitions with too many square changes."""
    timeline = FENTimeline(validate_transitions=True)
    
    # Valid starting position
    start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    assert timeline.collect(start) == start
    
    # Valid: e2-e4 (2 squares change)
    after_e4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    assert timeline.collect(after_e4) == after_e4
    
    # Invalid: jump to completely different position (many squares changed)
    random_pos = "8/8/8/3k4/8/8/8/4K3 w - - 0 1"
    assert timeline.collect(random_pos) is None
