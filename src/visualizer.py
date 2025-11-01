"""
Visualization utilities for drawing on video frames
"""

import cv2
import numpy as np

class Visualizer:
    def __init__(self, config):
        """
        Initialize visualizer
        
        Args:
            config: Visualization configuration dict
        """
        self.config = config
        self.trajectory_color = tuple(config['trajectory_color'])
        self.ball_color = tuple(config['ball_color'])
        self.thickness = config['line_thickness']
        
    def draw(self, frame, detection, trajectory):
        """
        Draw detection and trajectory on frame
        
        Args:
            frame: Input frame
            detection: Current detection dict or None
            trajectory: List of (x, y) positions
            
        Returns:
            Annotated frame
        """
        frame = frame.copy()
        
        # Draw trajectory
        if self.config['show_trajectory'] and len(trajectory) > 1:
            for i in range(1, len(trajectory)):
                cv2.line(frame, trajectory[i-1], trajectory[i], 
                        self.trajectory_color, self.thickness)
        
        # Draw current detection
        if detection is not None:
            x, y = detection['position']
            
            # Draw bounding box
            if self.config['show_bbox']:
                x1, y1, w, h = detection['bbox']
                cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), 
                             self.ball_color, self.thickness)
            
            # Draw center point
            cv2.circle(frame, (x, y), 8, self.ball_color, -1)
            
            # Draw confidence
            conf_text = f"{detection['confidence']:.2f}"
            cv2.putText(frame, conf_text, (x + 10, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.ball_color, 2)
        
        return frame
