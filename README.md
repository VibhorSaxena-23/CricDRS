# Cricket DRS - Ball Tracking System

Single-camera vision-based Decision Review System for cricket.

## Features

- ✅ Real-time ball detection using YOLOv8
- ✅ Ball tracking across frames
- ✅ Trajectory visualization
- ✅ Velocity calculation
- 🚧 DRS decision making (coming soon)

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Model

Place your trained YOLOv8 model in:
```
data/models/ball_detector.pt
```

### 3. Add Test Video

Place cricket video in:
```
data/input_video/
```

## Usage
```bash
python main.py
```

Output will be saved to data/output/

## Project Structure
```
CricDRS/
├── data/
│   ├── input_video/    # Input videos
│   ├── models/         # YOLO models
│   └── output/         # Processed videos
├── src/
│   ├── detector.py     # Ball detection
│   ├── tracker.py      # Position tracking
│   ├── visualizer.py   # Visualization
│   └── trajectory.py   # Trajectory analysis
├── utils/
│   └── video_utils.py  # Video I/O utilities
├── config.yml          # Configuration
├── main.py            # Main pipeline
└── requirements.txt   # Dependencies
```

## Configuration

Edit config.yml to adjust:
- Model path and confidence threshold
- Video input/output directories
- Tracker settings
- Visualization options

## Requirements

- Python 3.8+
- OpenCV
- PyTorch
- Ultralytics (YOLOv8)

## License

MIT License
