# # Chess Video to PGN Converter

Automatically converts chess game videos into PGN (Portable Game Notation) format using computer vision and deep learning.

## 📋 Overview

This tool processes chess game videos and extracts:
- Board detection and perspective correction
- Position recognition (FEN notation)
- Move inference
- PGN output with metadata

Inspired by [ChessCam.net](https://chesscam.net/) and [LiveChess2FEN](https://github.com/davidmallasen/LiveChess2FEN).

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repo-url>
cd chessPGN

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Process a single video
python main.py data/videos/game_1.mp4

# With custom output path
python main.py data/videos/game_1.mp4 -o output/game1.pgn

# Process with metadata
python main.py data/videos/game_1.mp4 \
  --white "Magnus Carlsen" \
  --black "Hikaru Nakamura" \
  --event "Speed Chess Championship"
```

### Advanced Options

```bash
# Force manual corner selection (if auto-detection fails)
python main.py data/videos/game_1.mp4 --manual

# Sample every 60 frames (for faster processing)
python main.py data/videos/game_1.mp4 -s 60

# Limit to first 20 frames (for testing)
python main.py data/videos/game_1.mp4 -m 20

# Disable debug visualizations
python main.py data/videos/game_1.mp4 --no-debug
```

## 📂 Project Structure

```
chessPGN/
├── data/
│   ├── videos/          # Input chess game videos
│   └── frames/          # Extracted frames (optional)
├── src/
│   ├── video_reader.py       # Video frame extraction
│   ├── preprocess.py         # Image preprocessing
│   ├── corner_finder.py      # Automatic board detection
│   ├── corner_picker.py      # Manual corner selection
│   ├── board_detector.py     # Board detection (alternative)
│   ├── fen_extractor.py      # Position recognition (FEN)
│   ├── move_inference.py     # Move sequence inference
│   └── pgn_writer.py         # PGN file generation
├── main.py              # Main pipeline
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🔧 Pipeline Architecture

### 1. Video Reading
- Samples frames at configurable intervals
- Default: every 30 frames (~1 second at 30fps)

### 2. Board Detection
**Automatic Detection (CameraChessWeb-inspired):**
- Shi-Tomasi corner detection
- Delaunay triangulation for grid inference
- Perspective transform scoring
- Falls back to manual if fails

**Manual Selection:**
- Interactive corner picking
- Validation checks (area, uniqueness)
- Auto-ordering (TL, TR, BR, BL)

### 3. Perspective Correction
- Homography computation
- Warps board to 800×800 square view

### 4. FEN Extraction
- **TODO:** Implement CNN-based piece recognition
- Current: Placeholder implementation
- Recommended: Fine-tuned CNN or YOLO model

### 5. Move Inference
- **TODO:** Implement chess engine integration
- Compares consecutive FEN positions
- Validates legal moves
- Handles ambiguous cases

### 6. PGN Generation
- Standard PGN format with Seven Tag Roster
- Movetext with proper numbering
- Result annotation

## 🎯 Current Status

### ✅ Implemented
- [x] Video frame extraction
- [x] Image preprocessing (cropping, resizing)
- [x] Automatic board corner detection
- [x] Manual corner selection fallback
- [x] Perspective correction
- [x] PGN file writing
- [x] Command-line interface

### 🚧 In Progress / TODO
- [ ] **FEN Extraction:** Implement piece recognition
  - Option 1: Train custom CNN on chess piece dataset
  - Option 2: Use pre-trained model (e.g., from LiveChess2FEN)
  - Option 3: Classical CV (template matching, color analysis)
  
- [ ] **Move Inference:** Implement position comparison
  - Use python-chess library
  - Validate legal moves
  - Handle castling, en passant, promotions
  
- [ ] **Robustness Improvements:**
  - Better lighting normalization
  - Handle different board styles/colors
  - Piece occlusion handling
  - Player hand detection (to avoid)

## 📊 Model Approach (Planned)

### Piece Recognition Options

**Option A: Deep Learning (Recommended)**
```
Model: Custom CNN or ResNet-18
Input: 64×64 RGB patches (each square)
Output: 13 classes (6 pieces × 2 colors + empty)
Training: Chess piece dataset (~50k images)
Accuracy Target: >95%
```

**Option B: Classical CV (Fallback)**
```
- Color segmentation (dark/light pieces)
- Template matching
- Contour analysis for piece shapes
- Less robust but no training needed
```

### Move Validation
```python
import chess

def infer_move(fen1: str, fen2: str) -> str:
    board = chess.Board(fen1)
    # Find legal move that results in fen2
    for move in board.legal_moves:
        board.push(move)
        if board.fen() == fen2:
            return move.uci()
        board.pop()
    return None
```

## 🧪 Testing

```bash
# Test video reader
python src/video_reader.py

# Test board detection on single frame
python -c "
import cv2
from src.corner_finder import find_board_corners
frame = cv2.imread('data/frames/test.jpg')
corners, debug = find_board_corners(frame, debug=True)
cv2.imshow('Debug', debug)
cv2.waitKey(0)
"
```

## 📝 Dependencies

### Core
- `opencv-python>=4.7.0` - Computer vision operations
- `numpy>=1.24.0` - Numerical computations

### Recommended (for full pipeline)
- `python-chess>=1.999` - Chess logic and validation
- `torch>=2.0.0` - Deep learning (if using CNN)
- `torchvision>=0.15.0` - Pre-trained models
- `pillow>=9.0.0` - Additional image processing

### Optional
- `matplotlib>=3.7.0` - Visualization
- `jupyter>=1.0.0` - Interactive development

## 🎥 Example Output

**Input:** `data/videos/game_1.mp4` (3-minute game)

**Output PGN:**
```pgn
[Event "Chess Game Video Analysis"]
[Site "Unknown"]
[Date "2025.01.25"]
[Round "?"]
[White "Unknown"]
[Black "Unknown"]
[Result "*"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. c3 Nf6 5. d4 exd4 6. cxd4 Bb4+ 7. Nc3 Nxe4
8. O-O Bxc3 9. d5 Ne5 10. bxc3 Nxc4 11. Qd4 Ncd6 *
```

## 🐛 Known Issues

1. **Corner Detection Sensitivity**
   - May fail with extreme angles (>45°)
   - Solution: Use manual selection or improve preprocessing

2. **Lighting Variations**
   - Dark/bright videos affect detection
   - Solution: Add adaptive histogram equalization

3. **FEN Extraction Placeholder**
   - Currently returns empty board
   - Solution: Implement piece recognition

## 🤝 Contributing

1. Implement FEN extraction (high priority)
2. Add move inference logic
3. Improve board detection robustness
4. Add unit tests
5. Create example videos and expected outputs

## 📄 License

MIT License - See LICENSE file

## 🙏 Acknowledgments

- [CameraChessWeb](https://github.com/davidmallasen/CameraChessWeb) - Board detection algorithm
- [LiveChess2FEN](https://github.com/davidmallasen/LiveChess2FEN) - FEN extraction approach
- [python-chess](https://github.com/niklasf/python-chess) - Chess logic library

## 📧 Contact

For questions or issues, please open a GitHub issue.