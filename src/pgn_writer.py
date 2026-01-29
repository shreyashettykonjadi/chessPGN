"""
PGN Writer - Generate PGN files from chess moves.

Supports standard PGN format with Seven Tag Roster and move notation.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import chess  # pyright: ignore[reportMissingImports]
import chess.pgn # pyright: ignore[reportMissingImports]


def write_pgn(
    moves: List[str],
    output_path: str,
    *,
    event: str = "Unknown",
    site: str = "Unknown",
    date: Optional[str] = None,
    round: str = "?",
    white: str = "Player 1",
    black: str = "Player 2",
    result: str = "*",
    additional_tags: Optional[Dict[str, str]] = None
) -> None:
    """
    Write a PGN file from a list of UCI moves.
    
    Args:
        moves: List of UCI move strings (e.g., ["e2e4", "e7e5"])
        output_path: Path to output PGN file
        event: Event name (default: "Unknown")
        site: Site name (default: "Unknown")
        date: Date string in YYYY.MM.DD format (default: today)
        round: Round number (default: "?")
        white: White player name (default: "Player 1")
        black: Black player name (default: "Player 2")
        result: Game result: "1-0", "0-1", "1/2-1/2", or "*" (default: "*")
        additional_tags: Optional dict of additional PGN tags
    """
    if date is None:
        date = datetime.now().strftime("%Y.%m.%d")
    
    # Create a new game
    game = chess.pgn.Game()
    
    # Set Seven Tag Roster
    game.headers["Event"] = event
    game.headers["Site"] = site
    game.headers["Date"] = date
    game.headers["Round"] = round
    game.headers["White"] = white
    game.headers["Black"] = black
    game.headers["Result"] = result
    
    # Add additional tags if provided
    if additional_tags:
        for key, value in additional_tags.items():
            game.headers[key] = value
    
    # Replay moves on the board
    node = game
    for uci_move in moves:
        try:
            move = chess.Move.from_uci(uci_move)
            if move in node.board().legal_moves:
                node = node.add_variation(move)
            else:
                # Skip illegal moves
                print(f"[pgn-writer] Warning: Illegal move {uci_move}, skipping")
        except ValueError:
            # Skip invalid UCI moves
            print(f"[pgn-writer] Warning: Invalid UCI move {uci_move}, skipping")
    
    # Write PGN file
    with open(output_path, "w", encoding="utf-8") as f:
        print(game, file=f, end="\n\n")
    
    print(f"[pgn-writer] Wrote {len(moves)} moves to {output_path}")


def write_pgn_from_fens(
    fen_timeline: List[str],
    output_path: str,
    *,
    event: str = "Unknown",
    site: str = "Unknown",
    date: Optional[str] = None,
    round: str = "?",
    white: str = "Player 1",
    black: str = "Player 2",
    result: str = "*",
    debug: bool = False
) -> int:
    """
    Write PGN file by inferring moves from FEN timeline.
    
    Args:
        fen_timeline: List of FEN strings
        output_path: Path to output PGN file
        event, site, date, round, white, black, result: PGN headers
        debug: Enable debug output
        
    Returns:
        Number of moves successfully written
    """
    from .move_inference import infer_moves_from_fens
    
    moves = infer_moves_from_fens(fen_timeline, debug=debug, allow_ambiguous=True)
    
    if not moves:
        print("[pgn-writer] No moves inferred from FEN timeline")
        return 0
    
    # Convert chess.Move objects to UCI strings
    move_strings = [move.uci() for move in moves]
    
    write_pgn(
        move_strings,
        output_path,
        event=event,
        site=site,
        date=date,
        round=round,
        white=white,
        black=black,
        result=result
    )
    
    return len(move_strings)
