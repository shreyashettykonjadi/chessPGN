from typing import Dict, Optional, Tuple, List
from collections import defaultdict
import chess

class FirstMoveTracker:
    """
    Delta-only first-move tracker.

    Rules enforced externally:
      - source empty >= min_empty_frames
      - dest occupied >= stability_frames
      - candidate emitted only when exactly one source & one dest
      - hand_present must be False (checked by caller)
    """

    def __init__(self, stability_frames: int = 8, min_empty_frames: int = 5, debug: bool = False):
        self.stability_frames = stability_frames
        self.min_empty_frames = min_empty_frames
        self.debug = debug

        # occupancy streaks per square
        self.empty_streak: Dict[str, int] = defaultdict(int)
        self.occ_streak: Dict[str, int] = defaultdict(int)
        # last observed occupancy
        self.last_occ: Dict[str, bool] = {}
        # whether we've emitted a candidate (one-shot)
        self.emitted = False

    def process_frame(self, frame_idx: int, detected: Dict[str, Optional[str]], hand_present: bool) -> Optional[Tuple[str, str]]:
        """
        Args:
            detected: mapping square -> label or None
            hand_present: whether a hand is visible this frame

        Returns:
            (from_sq, to_sq) as strings when a candidate is confirmed, else None
        """
        if self.emitted:
            return None

        # build occupancy boolean for all squares a1..h8
        occ: Dict[str, bool] = {}
        for sq_idx in range(64):
            sq = chess.square_name(sq_idx)
            occ[sq] = bool(detected.get(sq))

        # update streaks
        for sq, is_occ in occ.items():
            prev = self.last_occ.get(sq)
            if prev is None:
                # init
                self.last_occ[sq] = is_occ
                if is_occ:
                    self.occ_streak[sq] = 1
                    self.empty_streak[sq] = 0
                else:
                    self.empty_streak[sq] = 1
                    self.occ_streak[sq] = 0
                continue

            if is_occ:
                self.occ_streak[sq] = self.occ_streak.get(sq, 0) + 1
                self.empty_streak[sq] = 0
            else:
                self.empty_streak[sq] = self.empty_streak.get(sq, 0) + 1
                self.occ_streak[sq] = 0

            self.last_occ[sq] = is_occ

        if hand_present:
            # do not confirm while hand visible
            return None

        # find candidate sources and dests
        sources: List[str] = []
        dests: List[str] = []
        for sq in occ:
            if self.empty_streak.get(sq, 0) >= self.min_empty_frames:
                # require that it was occupied previously at least once before becoming empty
                # we use last_occ==False means empty now but we can't check prior; accept empties as candidate sources
                # caller must ensure conservative thresholds
                sources.append(sq)
            if self.occ_streak.get(sq, 0) >= self.stability_frames:
                dests.append(sq)

        # require exactly one source and one dest and that they differ
        if len(sources) == 1 and len(dests) == 1 and sources[0] != dests[0]:
            self.emitted = True
            if self.debug:
                print(f"[tracker] candidate emitted frame={frame_idx} src={sources[0]} dst={dests[0]}")
            return (sources[0], dests[0])

        return None
