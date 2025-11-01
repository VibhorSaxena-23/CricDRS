"""
Trajectory analysis and velocity calculation
"""

import numpy as np
from collections import deque

class TrajectoryAnalyzer:
    def __init__(self, smooth_window=5):
        """
        Initialize trajectory analyzer
        
        Args:
            smooth_window: Window size for smoothing
        """
        self.smooth_window = smooth_window
        self.positions = deque(maxlen=100)
        self.timestamps = deque(maxlen=100)
    
    def add_position(self, position, timestamp):
        """Add new position"""
        if position is not None:
            self.positions.append(position)
            self.timestamps.append(timestamp)
    
    def get_velocity(self):
        """Calculate velocity"""
        if len(self.positions) < 2:
            return 0.0
        
        p1 = np.array(self.positions[-2])
        p2 = np.array(self.positions[-1])
        dt = self.timestamps[-1] - self.timestamps[-2]
        
        if dt == 0:
            return 0.0
        
        distance = np.linalg.norm(p2 - p1)
        return distance / dt
    
    def fit_trajectory(self, degree=2):
        """Fit polynomial to trajectory"""
        if len(self.positions) < degree + 1:
            return None
        
        points = np.array(list(self.positions))
        x = points[:, 0]
        y = points[:, 1]
        
        try:
            coeffs = np.polyfit(x, y, degree)
            return coeffs
        except:
            return None
