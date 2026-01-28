from typing import Optional, List, Tuple, Dict, Any
from .fen_transition_validator import FENTransitionValidator


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

    def collect(self, fen: Optional[str]) -> Optional[str]:
        if fen is None:
            return None

        last_fen = self._entries[-1][0] if self._entries else None
        if fen == last_fen:
            return None

        if self._validate:
            valid, reason, meta = self._validator.validate(fen)
            if not valid:
                if self._debug:
                    print(f"[fen-timeline] Rejected FEN: {reason}")
                return None
            # accepted: store with move meta (may be None for first)
            self._entries.append((fen, meta))
            return fen
        else:
            # no validation: append without move meta
            self._entries.append((fen, None))
            return fen

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

    def __len__(self) -> int:
        return len(self._entries)
