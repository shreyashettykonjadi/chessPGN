import re
import logging
from typing import Dict, Optional, List
import chess
import chess.pgn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vision_interface")

SQUARE_RE = re.compile(r"^[a-h][1-8]$")
PIECE_SET = set(list("PNBRQKpnbrqk"))


class VisionInterface:
    """
    Encapsulates vision -> chess interface.
    Holds canonical chess.Board() and previous_piece_map lifecycle.
    """

    def __init__(self) -> None:
        # Canonical authoritative board (standard start) created once per instance
        self.board = chess.Board()
        # Log board initialization exactly once
        self.logger = logger
        self.logger.info("Canonical chess.Board initialized. White to move=%s", self.board.turn == chess.WHITE)
        # Module-level previous map moved to instance state
        self.previous_piece_map: Dict[str, Optional[str]] = {}
        # Temporal confirmation counters for legal moves
        self.move_counters: Dict[str, int] = {}
        self.N = 3  # Consecutive frames required for confirmation
        # Committed moves (extensible for multi-move)
        self.first_committed_move: Optional[str] = None

    def validate_piece_map(self, piece_map: Dict[str, Optional[str]]) -> bool:
        """
        Validate that piece_map is a dict mapping algebraic squares to single-letter piece codes.
        Example valid entry: { "e2": "P", "e7": "p", "g1": "N" }
        Values may be None to indicate 'empty' (optional).
        """
        if not isinstance(piece_map, dict):
            self.logger.error("piece_map must be a dict")
            return False

        for k, v in piece_map.items():
            if not isinstance(k, str) or not SQUARE_RE.match(k):
                self.logger.error("Invalid square key: %r", k)
                return False
            if v is None:
                continue
            if not isinstance(v, str) or len(v) != 1 or v not in PIECE_SET:
                self.logger.error("Invalid piece value at %s: %r", k, v)
                return False
        return True

    def write_pgn(self, output_path: str) -> None:
        """
        Write the current board state to a PGN file.
        Assumes moves have already been pushed to self.board.
        """
        game = chess.pgn.Game.from_board(self.board)
        game.headers["Event"] = "Chess Game"
        game.headers["Site"] = "Video"
        game.headers["White"] = "White"
        game.headers["Black"] = "Black"
        game.headers["Result"] = "*"

        with open(output_path, "w") as f:
            f.write(str(game))
        self.logger.info("PGN written to: %s", output_path)

    def process_frame(self, piece_map: Dict[str, Optional[str]]) -> None:
        """
        Entry point for vision per-frame output.
        - Validates piece_map format.
        - Computes appeared / disappeared squares vs previous frame.
        - Logs results and updates previous_piece_map.
        """
        ok = self.validate_piece_map(piece_map)
        if not ok:
            self.logger.warning("Received invalid piece_map; ignoring for now")
            return

        # Build sets of squares that are considered "present" (value not None)
        curr_keys = {k for k, v in piece_map.items() if v is not None}
        prev_keys = {k for k, v in self.previous_piece_map.items() if v is not None}

        appeared = sorted(curr_keys - prev_keys)
        disappeared = sorted(prev_keys - curr_keys)

        self.logger.info(
            "Frame delta: appeared=%d disappeared=%d", len(appeared), len(disappeared)
        )
        if appeared:
            self.logger.info("Appeared squares: %s", ", ".join(appeared))
        if disappeared:
            self.logger.info("Disappeared squares: %s", ", ".join(disappeared))

        # STEP 3 — candidate move generation
        candidate_moves = []
        for from_sq in disappeared:
            for to_sq in appeared:
                candidate_moves.append(from_sq + to_sq)

        self.logger.info("Generated %d candidate moves: %s",
                         len(candidate_moves), candidate_moves)

        # STEP 4 — candidate move validation (NO board mutation)
        legal_candidates: List[str] = []
        for uci in candidate_moves:
            try:
                move = chess.Move.from_uci(uci)
            except ValueError:
                self.logger.info("Candidate move %s → REJECTED (invalid UCI)", uci)
                continue

            if self.board.is_legal(move):
                self.logger.info("Candidate move %s → ACCEPTED (legal)", uci)
                legal_candidates.append(uci)
            else:
                self.logger.info("Candidate move %s → REJECTED (illegal)", uci)

        # STEP 5 — temporal confirmation (increment counters for legal moves seen this frame)
        seen_this_frame = set(legal_candidates)
        # Reset counters only if different legal moves are seen this frame
        if seen_this_frame:
            for uci in list(self.move_counters.keys()):
                if uci not in seen_this_frame:
                    self.move_counters[uci] = 0
        # Increment counters for seen moves
        for uci in seen_this_frame:
            self.move_counters[uci] = self.move_counters.get(uci, 0) + 1
        # Log stable moves (reached N consecutive frames)
        for uci, count in self.move_counters.items():
            if count >= self.N:
                self.logger.info("STABLE: %s after %d consecutive frames", uci, count)
                # Commit the stable move to the board
                move = chess.Move.from_uci(uci)
                self.board.push(move)
                self.first_committed_move = uci
                self.logger.info("COMMITTED MOVE: %s", uci)
                # Write PGN after commit
                self.write_pgn("output.pgn")

        # Update previous map for next frame
        self.previous_piece_map.clear()
        self.previous_piece_map.update(piece_map)


# Module-level instance to preserve simple call sites
_VISION = VisionInterface()


def process_frame(piece_map: Dict[str, Optional[str]]) -> None:
    """Module-level delegate to VisionInterface.process_frame."""
    _VISION.process_frame(piece_map)
    logger.info("Processing frame from video")



