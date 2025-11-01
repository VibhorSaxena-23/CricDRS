"""
Main pipeline for cricket ball detection and tracking
Usage: python main.py
"""

import cv2
import yaml
from pathlib import Path
from src.detector import BallDetector
from src.tracker import BallTracker
from src.visualizer import Visualizer
from utils.video_utils import VideoProcessor

def load_config(config_path='config.yml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Load configuration
    config = load_config()
    
    # Initialize components
    print("Initializing detector...")
    detector = BallDetector(
        model_path=config['model']['path'],
        confidence=config['model']['confidence_threshold'],
        device=config['model']['device']
    )
    
    print("Initializing tracker...")
    tracker = BallTracker(max_history=config['tracker']['max_history'])
    
    print("Initializing visualizer...")
    visualizer = Visualizer(config['visualization'])
    
    # Get input video
    input_dir = Path(config['video']['input_dir'])
    video_files = list(input_dir.glob('*.mp4'))
    
    if not video_files:
        print(f"No video files found in {input_dir}")
        return
    
    # Process first video
    video_path = video_files[0]
    print(f"\nProcessing: {video_path.name}")
    
    # Setup video processor
    output_path = Path(config['video']['output_dir']) / f"tracked_{video_path.name}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    processor = VideoProcessor(str(video_path), str(output_path))
    
    # Process video
    frame_count = 0
    
    for frame in processor.read_frames():
        # Detect ball
        detection = detector.detect(frame)
        
        # Update tracker
        timestamp = frame_count / processor.fps
        tracker.update(detection, timestamp)
        
        # Visualize
        frame = visualizer.draw(frame, detection, tracker.get_trajectory())
        
        # Write output
        processor.write_frame(frame)
        
        frame_count += 1
        
        # Progress
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")
    
    processor.release()
    print(f"\n✅ Done! Output saved to: {output_path}")

if __name__ == "__main__":
    main()
