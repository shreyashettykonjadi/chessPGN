import re
import logging
from typing import Dict, Optional, List
from collections import Counter
import chess
import chess.pgn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vision_interface")

SQUARE_RE = re.compile(r"^[a-h][1-8]$")
PIECE_SET = set(list("PNBRQKpnbrqk"))
MIN_VALID_PIECES = 6
LABEL_MAP = {
    'white_pawn': 'P',
    'white_knight': 'N',
    'white_bishop': 'B',
    'white_rook': 'R',
    'white_queen': 'Q',
    'white_king': 'K',
    'black_pawn': 'p',
    'black_knight': 'n',
    'black_bishop': 'b',
    'black_rook': 'r',
    'black_queen': 'q',
    'black_king': 'k',
}

# STEP B: Narrow delta scope to pawn starts only
TRACKED_SQUARES = {"e2", "e4", "d2", "d4"}


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
        # STEP C: First-move aggressive commit state
        self.first_move_committed = False
        self.pawn_move_votes: Counter = Counter()
        # Debug flag for verbose logging
        self.debug = False

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

    def process_frame(self, piece_map: Dict[str, Optional[str]], frame_idx: int) -> None:
        """
        Entry point for vision per-frame output.
        - Validates piece_map format.
        - Computes appeared / disappeared squares vs previous frame.
        - Logs results and updates previous_piece_map.
        """
        # If first move already committed, skip all processing
        if self.first_move_committed:
            return

        # TEMP: STEP 3 LABEL MAP
        # Canonicalize labels: map model labels to chess symbols, drop unmapped
        canonicalized = {}
        mapped_count = 0
        dropped_count = 0
        for sq, lbl in piece_map.items():
            if lbl in LABEL_MAP:
                canonicalized[sq] = LABEL_MAP[lbl]
                mapped_count += 1
            else:
                dropped_count += 1
        self.logger.info("[canonicalized] frame=%d mapped=%d dropped=%d", frame_idx, mapped_count, dropped_count)

        # TEMP: STEP 1 SANITIZATION (relaxed: filter invalid entries instead of rejecting frame)
        original_count = len(canonicalized)
        sanitized = {}
        for k, v in canonicalized.items():
            if not isinstance(k, str) or not SQUARE_RE.match(k):
                continue  # drop invalid square
            if v is not None and (not isinstance(v, str) or v not in PIECE_SET):
                continue  # drop invalid piece
            sanitized[k] = v
        kept = len(sanitized)
        dropped = original_count - kept
        if self.debug:
            self.logger.info("[sanitized] frame=%d kept=%d dropped=%d", frame_idx, kept, dropped)

        # STEP A: The ONLY hard reject — if kept < 4
        if kept < 4:
            self.logger.warning("Frame %d rejected: insufficient pieces for delta (%d < 4)", frame_idx, kept)
            return

        # STEP A: Sanity is LOGGING ONLY, NOT control flow
        white_pieces = sum(1 for v in sanitized.values() if isinstance(v, str) and v.isupper())
        black_pieces = sum(1 for v in sanitized.values() if isinstance(v, str) and v.islower())
        white_kings = sum(1 for v in sanitized.values() if v == "K")
        black_kings = sum(1 for v in sanitized.values() if v == "k")
        if self.debug:
            if white_kings != 1:
                self.logger.warning("Board sanity warning (ignored): white_kings=%d", white_kings)
            if black_kings != 1:
                self.logger.warning("Board sanity warning (ignored): black_kings=%d", black_kings)
            if white_pieces > 16:
                self.logger.warning("Board sanity warning (ignored): white_piece_count=%d", white_pieces)
            if black_pieces > 16:
                self.logger.warning("Board sanity warning (ignored): black_piece_count=%d", black_pieces)
        # Processing continues regardless of sanity

        # STEP A: Remove piece count gating — delta runs if kept >=4, regardless of count
        # (Removed: if not (28 <= curr_count <= 32): skip delta)

        # STEP B: Filter to TRACKED_SQUARES only
        curr_tracked = {sq: sanitized.get(sq) for sq in TRACKED_SQUARES}
        prev_tracked = {sq: self.previous_piece_map.get(sq) for sq in TRACKED_SQUARES}

        # Build sets of squares that have a pawn present (value == 'P')
        curr_pawn_squares = {sq for sq, v in curr_tracked.items() if v == 'P'}
        prev_pawn_squares = {sq for sq, v in prev_tracked.items() if v == 'P'}

        appeared = sorted(curr_pawn_squares - prev_pawn_squares)
        disappeared = sorted(prev_pawn_squares - curr_pawn_squares)

        self.logger.info("[frame-delta-tracked] frame=%d appeared=%s disappeared=%s", frame_idx, appeared, disappeared)

        # STEP B/C: Detect pawn moves (e2→e4 or d2→d4)
        if len(disappeared) == 1 and len(appeared) == 1:
            from_sq = disappeared[0]
            to_sq = appeared[0]
            # Valid pawn push patterns for first move
            if (from_sq == "e2" and to_sq == "e4") or (from_sq == "d2" and to_sq == "d4"):
                move_key = f"{from_sq}{to_sq}"
                self.pawn_move_votes[move_key] += 1
                self.logger.info("[first-move-tracking] candidate=%s frame=%d count=%d", move_key, frame_idx, self.pawn_move_votes[move_key])

                # STEP C: Commit when vote reaches N
                if self.pawn_move_votes[move_key] >= self.N:
                    self.logger.info("[first-move-commit] move=%s", move_key)
                    # Commit the stable move to the board
                    move = chess.Move.from_uci(move_key)
                    if self.board.is_legal(move):
                        self.board.push(move)
                        self.first_committed_move = move_key
                        self.first_move_committed = True
                        self.logger.info("Frame %d COMMITTED MOVE: %s", frame_idx, move_key)
                        # Write PGN after commit
                        self.write_pgn("output.pgn")
                        # Reset all counters
                        self.pawn_move_votes.clear()
                        self.move_counters.clear()
                    else:
                        self.logger.warning("Frame %d move %s is NOT legal per python-chess", frame_idx, move_key)
        else:
            # No valid single pawn move detected — reset votes for interrupted sequences
            # But only reset if we see a contradicting pattern
            pass

        # Update previous map for next frame (full sanitized, not just tracked)
        self.previous_piece_map.clear()
        self.previous_piece_map.update(sanitized)


# Module-level instance to preserve simple call sites
_VISION = VisionInterface()


def process_frame(piece_map: Dict[str, Optional[str]], frame_idx: int) -> None:
    """Module-level delegate to VisionInterface.process_frame."""
    _VISION.process_frame(piece_map, frame_idx)
    if _VISION.debug:
        logger.info("Processing frame from video (frame %d)", frame_idx)



