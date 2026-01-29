"""
Stable Snapshot Move Detector

Detects chess moves by comparing stable board snapshots.
A snapshot is stable if the same square→piece mapping appears for N consecutive frames.
"""

from typing import Dict, Optional, Tuple, List
import hashlib
import chess  # pyright: ignore[reportMissingImports]


def hash_piece_map(piece_map: Dict[str, str]) -> str:
    """Generate a hash of a piece map for comparison."""
    # Sort items for consistent hashing
    sorted_items = sorted(piece_map.items())
    content = str(sorted_items)
    return hashlib.md5(content.encode()).hexdigest()[:8]


def piece_map_to_fen_board(piece_map: Dict[str, str]) -> str:
    """
    Convert piece map {square: piece_label} to FEN board string.
    
    Piece labels expected: "white_pawn", "black_rook", etc.
    FEN chars: P, R, N, B, Q, K (white) / p, r, n, b, q, k (black)
    """
    PIECE_TO_FEN = {
        "white_pawn": "P", "white_rook": "R", "white_knight": "N",
        "white_bishop": "B", "white_queen": "Q", "white_king": "K",
        "black_pawn": "p", "black_rook": "r", "black_knight": "n",
        "black_bishop": "b", "black_queen": "q", "black_king": "k",
    }
    
    FILES = ["a", "b", "c", "d", "e", "f", "g", "h"]
    rows = []
    
    # FEN ranks go from 8 -> 1
    for rank in range(8, 0, -1):
        empty = 0
        row = ""
        
        for file in FILES:
            square = f"{file}{rank}"
            if square in piece_map:
                if empty > 0:
                    row += str(empty)
                    empty = 0
                label = piece_map[square]
                fen_char = PIECE_TO_FEN.get(label, "?")
                row += fen_char
            else:
                empty += 1
        
        if empty > 0:
            row += str(empty)
        
        rows.append(row)
    
    return "/".join(rows)


class StableSnapshotDetector:
    """
    Detects stable board snapshots and infers moves from stable→stable transitions.
    """
    
    def __init__(self, stability_threshold: int = 5, debug: bool = True):
        """
        Args:
            stability_threshold: Number of consecutive frames required for stability (default: 5)
            debug: Enable debug output
        """
        self.stability_threshold = stability_threshold
        self.debug = debug
        
        # Buffer for recent piece maps
        self._recent_maps: List[Tuple[int, Dict[str, str], str]] = []  # (frame_idx, piece_map, hash)
        
        # Last stable snapshot
        self._last_stable: Optional[Tuple[int, Dict[str, str], str]] = None
        
        # Persistent chess board
        self._board = chess.Board()
        
        # Detected moves (UCI strings)
        self._detected_moves: List[str] = []
    
    def process_frame(self, frame_idx: int, piece_map: Dict[str, str]) -> Optional[str]:
        """
        Process a frame's piece map.
        
        Args:
            frame_idx: Frame index
            piece_map: Dict mapping square -> piece_label (e.g., {"e2": "white_pawn"})
        
        Returns:
            UCI move string if a move was detected, None otherwise
        """
        # Hash the piece map
        map_hash = hash_piece_map(piece_map)
        
        # Add to recent buffer
        self._recent_maps.append((frame_idx, piece_map.copy(), map_hash))
        
        # Keep only recent maps (need at least stability_threshold)
        if len(self._recent_maps) > self.stability_threshold * 2:
            self._recent_maps.pop(0)
        
        # Check if we have enough frames for stability check
        if len(self._recent_maps) < self.stability_threshold:
            # Silent - building buffer
            return None
        
        # Check if last N frames have the same hash (or majority match)
        recent_hashes = [h for _, _, h in self._recent_maps[-self.stability_threshold:]]
        unique_hashes = len(set(recent_hashes))
        
        # Count occurrences of each hash
        hash_counts = {}
        for h in recent_hashes:
            hash_counts[h] = hash_counts.get(h, 0) + 1
        
        # Find most common hash
        most_common_hash = max(hash_counts.items(), key=lambda x: x[1])[0] if hash_counts else None
        most_common_count = hash_counts.get(most_common_hash, 0) if most_common_hash else 0
        
        # Consider stable if most common hash appears in at least (threshold - 1) frames
        # This allows for 1 frame of noise
        is_stable = most_common_count >= (self.stability_threshold - 1) and most_common_hash == map_hash
        
        # Only log when stable snapshot is detected (not every frame)
        if not is_stable:
            return None
        
        # Stable snapshot detected! Use the most common hash's map
        # Find the frame with the most common hash (we already computed it above)
        stable_entry = None
        for entry in reversed(self._recent_maps[-self.stability_threshold:]):
            if entry[2] == most_common_hash:
                stable_entry = entry
                break
        
        if stable_entry is None:
            stable_entry = self._recent_maps[-1]  # Fallback to most recent
        
        stable_map = stable_entry[1]
        stable_hash = stable_entry[2]
        stable_frame = stable_entry[0]
        
        # Compare with last stable snapshot
        if self._last_stable is None:
            # First stable snapshot - initialize
            # Diagnostic: compare detected map vs canonical start (visibility only; no filtering)
            try:
                PIECE_TO_FEN = {
                    "white_pawn": "P", "white_rook": "R", "white_knight": "N",
                    "white_bishop": "B", "white_queen": "Q", "white_king": "K",
                    "black_pawn": "p", "black_rook": "r", "black_knight": "n",
                    "black_bishop": "b", "black_queen": "q", "black_king": "k",
                }
                canonical = chess.Board()
                conflicts = 0
                for sq_idx in range(64):
                    sq = chess.square_name(sq_idx)
                    expected_piece = canonical.piece_at(sq_idx)
                    expected_label = None
                    if expected_piece:
                        sym = expected_piece.symbol()
                        # convert symbol to our label space
                        label_map = {
                            'P': 'white_pawn', 'N': 'white_knight', 'B': 'white_bishop',
                            'R': 'white_rook', 'Q': 'white_queen', 'K': 'white_king',
                            'p': 'black_pawn', 'n': 'black_knight', 'b': 'black_bishop',
                            'r': 'black_rook', 'q': 'black_queen', 'k': 'black_king',
                        }
                        expected_label = label_map.get(sym)
                    observed_label = stable_map.get(sq)
                    if observed_label and expected_label and observed_label != expected_label:
                        conflicts += 1
                if self.debug:
                    print(f"[snapshot-init] frame={stable_frame} raw_detections_count={len(stable_map)} conflicts_vs_start={conflicts}")
            except Exception:
                pass
            self._last_stable = (stable_frame, stable_map, stable_hash)
            if self.debug:
                print(f"[stable-snapshot] Initial board detected (frame {stable_frame}, {len(stable_map)} pieces)")
            return None
        
        # Only log when comparing snapshots (not every stable detection)
        if self.debug:
            print(f"[stable-snapshot] Frame {stable_frame}: Comparing snapshots ({len(stable_map)} pieces)")
        
        # Compare snapshots
        last_map = self._last_stable[1]
        move = self._compare_snapshots(last_map, stable_map, stable_frame)
        
        if move:
            # Update last stable snapshot
            self._last_stable = (stable_frame, stable_map, stable_hash)
            self._detected_moves.append(move)
            return move
        
        # No move detected, but update last stable if hash changed
        if stable_hash != self._last_stable[2]:
            self._last_stable = (stable_frame, stable_map, stable_hash)
        
        return None
    
    def _compare_snapshots(
        self, 
        prev_map: Dict[str, str], 
        curr_map: Dict[str, str],
        frame_idx: int
    ) -> Optional[str]:
        """
        Compare two stable snapshots and infer move using chess.Board validation.
        More robust to detection noise by testing all legal moves and scoring them.
        
        Returns:
            UCI move string if valid move detected, None otherwise
        """
        # Find squares that differ
        all_squares = set(prev_map.keys()) | set(curr_map.keys())
        changed_squares = []
        
        for sq in all_squares:
            prev_piece = prev_map.get(sq)
            curr_piece = curr_map.get(sq)
            
            if prev_piece != curr_piece:
                changed_squares.append((sq, prev_piece, curr_piece))
        
        # Only show detailed changes if debug and there are reasonable changes (potential move)
        if self.debug and 1 <= len(changed_squares) <= 10:
            print(f"[stable-snapshot] Frame {frame_idx}: {len(changed_squares)} squares differ")
            for sq, prev, curr in changed_squares[:5]:  # Show up to 5 changes
                print(f"  {sq}: {prev} -> {curr}")
            if len(changed_squares) > 5:
                print(f"  ... and {len(changed_squares) - 5} more")
        
        # Skip if no changes
        if len(changed_squares) == 0:
            return None  # Silent - same snapshot
        
        # Skip if too many changes (likely detection noise, not a real move)
        if len(changed_squares) > 15:
            if self.debug:
                print(f"[stable-snapshot] Frame {frame_idx}: Too many changes ({len(changed_squares)}), skipping")
            return None
        
        # Build FEN boards from piece maps
        prev_fen_board = piece_map_to_fen_board(prev_map)
        curr_fen_board = piece_map_to_fen_board(curr_map)
        
        # Try all legal moves and score them
        legal_moves = list(self._board.legal_moves)
        best_match = None
        best_score = -1
        
        # Build a map of what changed: square -> (prev_piece, curr_piece)
        change_map = {sq: (prev, curr) for sq, prev, curr in changed_squares}
        changed_square_set = set(change_map.keys())
        
        for move in legal_moves:
            from_sq = chess.square_name(move.from_square)
            to_sq = chess.square_name(move.to_square)
            
            # Test this move
            self._board.push(move)
            result_fen_board = self._board.board_fen()
            self._board.pop()
            
            # Score: how well does this move explain the observed changes?
            score = 0
            
            # Get the piece that would be moving
            moving_piece = self._board.piece_at(move.from_square)
            if moving_piece is None:
                continue  # Skip if no piece at from square
            
            # Convert piece to our label format for comparison
            piece_color = "white" if moving_piece.color == chess.WHITE else "black"
            piece_type_map = {
                chess.PAWN: "pawn",
                chess.ROOK: "rook",
                chess.KNIGHT: "knight",
                chess.BISHOP: "bishop",
                chess.QUEEN: "queen",
                chess.KING: "king"
            }
            expected_piece_label = f"{piece_color}_{piece_type_map[moving_piece.piece_type]}"
            
            # CRITICAL: Check if from_square actually lost a piece (be more lenient with piece type)
            if from_sq in change_map:
                prev_piece, curr_piece = change_map[from_sq]
                # From square should have had a piece, now be empty
                if prev_piece == expected_piece_label and not curr_piece:
                    score += 35  # Perfect match: from square lost the correct piece
                elif prev_piece and not curr_piece:
                    # Lost some piece - check if same color (more lenient)
                    prev_color = prev_piece.split('_')[0] if '_' in prev_piece else ""
                    expected_color = expected_piece_label.split('_')[0] if '_' in expected_piece_label else ""
                    if prev_color == expected_color:
                        score += 25  # Same color piece lost (good match despite type noise)
                    else:
                        score += 15  # Different color piece lost (weaker match)
                elif prev_piece and curr_piece and prev_piece != curr_piece:
                    score += 5  # Piece changed (likely noise)
            
            # CRITICAL: Check if to_square actually gained a piece (be more lenient)
            if to_sq in change_map:
                prev_piece, curr_piece = change_map[to_sq]
                # To square should have been empty (or had opponent piece), now has a piece
                if not prev_piece and curr_piece == expected_piece_label:
                    score += 35  # Perfect match: to square gained the correct piece
                elif not prev_piece and curr_piece:
                    # Gained some piece - check if same color
                    curr_color = curr_piece.split('_')[0] if '_' in curr_piece else ""
                    expected_color = expected_piece_label.split('_')[0] if '_' in expected_piece_label else ""
                    if curr_color == expected_color:
                        score += 25  # Same color piece gained (good match despite type noise)
                    else:
                        score += 10  # Different color piece gained (weaker)
                elif prev_piece and curr_piece == expected_piece_label:
                    score += 25  # Capture: opponent piece replaced by our piece
                elif prev_piece and curr_piece:
                    # Piece changed - check if gained correct color
                    curr_color = curr_piece.split('_')[0] if '_' in curr_piece else ""
                    expected_color = expected_piece_label.split('_')[0] if '_' in expected_piece_label else ""
                    if curr_color == expected_color:
                        score += 15  # Same color piece appeared (possible capture or promotion)
            
            # Bonus: if both from and to match the pattern (more lenient)
            if from_sq in change_map and to_sq in change_map:
                from_prev, from_curr = change_map[from_sq]
                to_prev, to_curr = change_map[to_sq]
                # Perfect pattern: from had correct piece -> empty, to empty -> has correct piece
                if (from_prev == expected_piece_label and not from_curr and 
                    not to_prev and to_curr == expected_piece_label):
                    score += 25  # Perfect pattern match
                # Lenient pattern: from had piece -> empty, to empty -> has piece (same color)
                elif (from_prev and not from_curr and not to_prev and to_curr):
                    from_color = from_prev.split('_')[0] if '_' in from_prev else ""
                    to_color = to_curr.split('_')[0] if '_' in to_curr else ""
                    expected_color = expected_piece_label.split('_')[0] if '_' in expected_piece_label else ""
                    if from_color == expected_color and to_color == expected_color:
                        score += 15  # Good pattern match (same color, type may differ due to noise)
            
            # Compare result board with current board (more accurate comparison)
            # Count squares that match between result and current
            matches = 0
            mismatches = 0
            
            # Parse FEN boards to compare square by square
            def fen_to_square_map(fen_board: str) -> Dict[str, str]:
                """Convert FEN board to square->piece map."""
                square_map = {}
                ranks = fen_board.split('/')
                for rank_idx, rank in enumerate(ranks):
                    file_idx = 0
                    for char in rank:
                        if char.isdigit():
                            file_idx += int(char)
                        else:
                            sq = f"{chr(ord('a') + file_idx)}{8 - rank_idx}"
                            square_map[sq] = char
                            file_idx += 1
                return square_map
            
            result_squares = fen_to_square_map(result_fen_board)
            curr_squares = fen_to_square_map(curr_fen_board)
            
            # Compare squares
            all_check_squares = set(result_squares.keys()) | set(curr_squares.keys())
            for sq in all_check_squares:
                result_piece = result_squares.get(sq, '1')
                curr_piece = curr_squares.get(sq, '1')
                if result_piece == curr_piece:
                    matches += 1
                else:
                    mismatches += 1
            
            # Score based on match ratio (more matches = better)
            if matches + mismatches > 0:
                match_ratio = matches / (matches + mismatches)
                score += match_ratio * 30  # Up to 30 points for board similarity
            
            if score > best_score:
                best_score = score
                best_match = move
        
        # Accept move if score is high enough
        # Lower threshold to handle detection noise - prioritize moves where from/to squares match
        threshold = 35  # Lower threshold: from (25) + to (25) = 50, but allow lower for noise
        
        # Also accept if it's the only reasonable candidate and score is decent
        # (helps when detection is very noisy)
        if best_match and best_score >= 20 and len(changed_squares) <= 4:
            # If only a few squares changed and we have a reasonable match, be more lenient
            from_sq = chess.square_name(best_match.from_square)
            to_sq = chess.square_name(best_match.to_square)
            if from_sq in changed_square_set and to_sq in changed_square_set:
                # Both squares in changed set - likely a real move despite lower score
                threshold = 20
        
        if best_match and best_score >= threshold:
            # Final validation
            if best_match in self._board.legal_moves:
                # Get piece info before pushing
                expected_piece = self._board.piece_at(best_match.from_square)
                piece_str = str(expected_piece) if expected_piece else "?"
                
                self._board.push(best_match)
                if self.debug:
                    print(f"[stable-snapshot] ACCEPTED: {best_match.uci()} ({piece_str}) score={best_score:.1f}, frame {frame_idx}")
                return best_match.uci()
        
        # Only log rejections if debug and we had a candidate (suppress silent failures)
        if self.debug and best_match and best_score > 20:  # Only log if we had a reasonable candidate
            expected_piece = self._board.piece_at(best_match.from_square)
            piece_str = str(expected_piece) if expected_piece else "?"
            print(f"[stable-snapshot] Frame {frame_idx}: Rejected {best_match.uci()} ({piece_str}) - score {best_score:.1f} < {threshold}")
        
        return None
    
    def get_detected_moves(self) -> List[str]:
        """Return list of detected moves (UCI strings)."""
        return self._detected_moves.copy()
    
    def reset(self):
        """Reset detector state."""
        self._recent_maps.clear()
        self._last_stable = None
        self._board = chess.Board()
        self._detected_moves.clear()
