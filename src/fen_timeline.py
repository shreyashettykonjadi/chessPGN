from typing import Optional, List, Tuple, Dict, Any
from .fen_transition_validator import FENTransitionValidator
from collections import Counter


class FENTimeline:
    """
    FEN timeline with deduplication and transition validation.
    Internally stores list[ (fen: str, move_meta: Optional[dict]) ].
    """

    def __init__(self, validate_transitions: bool = True, debug: bool = False):
        self._entries: List[Tuple[str, Optional[Dict[str, Any]]]] = []
        self._validate = validate_transitions
        self._debug = debug
        self._validator = FENTransitionValidator(debug=debug)
        # Buffer for recently rejected candidates; do not advance baseline from these
        self._buffer: List[str] = []
        self._max_buffer: int = 5

        # Confirmation buffer for VALID candidates (post-move confirmation)
        # Reduced from 3 to 1 to be more responsive to moves in high FPS videos
        self._confirm_buffer: List[str] = []
        self._confirm_size: int = 1

    def collect(self, fen: Optional[str]) -> Optional[str]:
        if fen is None:
            return None

        last_fen = self._entries[-1][0] if self._entries else None
        if fen == last_fen:
            return None

        if not self._validate:
            # no validation: append without move meta
            self._entries.append((fen, None))
            return fen

        # With validation: attempt to validate candidate against validator's baseline.
        valid, reason, meta = self._validator.validate(fen)

        if not valid:
            # Rejected: buffer it (do not replace baseline)
            if self._debug:
                print(f"[fen-timeline] Rejected FEN (buffering): {reason} -> {fen}")

            # Avoid duplicates in buffer
            if not self._buffer or self._buffer[-1] != fen:
                self._buffer.append(fen)
                # enforce max buffer size
                if len(self._buffer) > self._max_buffer:
                    self._buffer.pop(0)
            # Reset any confirm buffer for VALID candidates (optional safety)
            # (we keep confirm buffer as-is to allow separate flows)
            return None

        # Candidate validated by validator: handle confirmation buffering
        # If there is no prior accepted entry, accept immediately (first baseline)
        if last_fen is None:
            self._entries.append((fen, meta))
            if self._debug:
                print(f"[fen-timeline] Accepted initial FEN: {fen} -> meta={meta}")
            # Clear rejected buffer when baseline moves to first accepted
            self._buffer.clear()
            return fen

        # Buffer validated candidate for confirmation instead of immediate accept
        if self._debug:
            print(f"[fen-timeline] Valid candidate buffered for confirmation: {fen}")

        # Avoid consecutive duplicate entries in confirmation buffer
        if not self._confirm_buffer or self._confirm_buffer[-1] != fen:
            self._confirm_buffer.append(fen)
            if len(self._confirm_buffer) > self._confirm_size:
                self._confirm_buffer.pop(0)

        # If not enough confirmations yet, wait
        if len(self._confirm_buffer) < self._confirm_size:
            if self._debug:
                print(f"[fen-timeline] Waiting for confirmations ({len(self._confirm_buffer)}/{self._confirm_size})")
            return None

        # Enough candidates collected: pick the most common candidate in confirm buffer
        counter = Counter(self._confirm_buffer)
        candidate_fen, _ = counter.most_common(1)[0]

        # Re-run validator on the chosen candidate to get fresh meta
        valid2, reason2, meta2 = self._validator.validate(candidate_fen)
        if valid2:
            # Accept candidate, clear buffers
            self._entries.append((candidate_fen, meta2))
            if self._debug:
                print(f"[fen-timeline] Accepted FEN after confirmation: {candidate_fen} -> meta={meta2}")
            self._confirm_buffer.clear()
            self._buffer.clear()
            return candidate_fen
        else:
            # Candidate failed on final check: drop oldest buffered candidate and keep waiting
            if self._debug:
                print(f"[fen-timeline] Candidate failed final validation: {reason2} -> {candidate_fen}")
            # pop oldest (FIFO) to make room for future candidates
            if self._confirm_buffer:
                self._confirm_buffer.pop(0)
            return None

    @property
    def history(self) -> List[str]:
        """Return list of FEN strings (backwards-compatible)."""
        return [fen for fen, _ in self._entries]

    def entries(self) -> List[Tuple[str, Optional[Dict[str, Any]]]]:
        """Return full entries with move metadata."""
        return self._entries.copy()

    @property
    def last_fen(self) -> Optional[str]:
        return self._entries[-1][0] if self._entries else None

    def reset(self):
        self._entries.clear()
        self._validator.reset()
        self._buffer.clear()
        self._confirm_buffer.clear()

    def __len__(self) -> int:
        return len(self._entries)
