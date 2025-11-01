"""
Simple ball position tracker
"""

from collections import deque
import numpy as np

class BallTracker:
    def __init__(self, max_history=50):
        """
        Initialize tracker
        
        Args:
            max_history: Maximum number of positions to store
        """
        self.positions = deque(maxlen=max_history)
        self.timestamps = deque(maxlen=max_history)
        self.confidences = deque(maxlen=max_history)
        
    def update(self, detection, timestamp):
        """
        Update tracker with new detection
        
        Args:
            detection: Detection dict or None
            timestamp: Current timestamp
        """
        if detection is not None:
            self.positions.append(detection['position'])
            self.timestamps.append(timestamp)
            self.confidences.append(detection['confidence'])
    
    def get_trajectory(self):
        """Get list of tracked positions"""
        return list(self.positions)
    
    def get_velocity(self):
        """Calculate current velocity in pixels/second"""
        if len(self.positions) < 2:
            return 0.0
        
        # Last two positions
        p1 = np.array(self.positions[-2])
        p2 = np.array(self.positions[-1])
        
        # Time difference
        dt = self.timestamps[-1] - self.timestamps[-2]
        
        if dt == 0:
            return 0.0
        
        # Distance
        distance = np.linalg.norm(p2 - p1)
        
        return distance / dt
    
    def is_tracking(self):
        """Check if actively tracking"""
        return len(self.positions) > 0
