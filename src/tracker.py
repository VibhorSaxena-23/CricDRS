"""
Enhanced Ball Position Tracker with prediction, validation, and quality metrics
"""
import numpy as np
from collections import deque
from enum import Enum


class TrackingState(Enum):
    """Tracking state enumeration"""
    IDLE = 0
    TRACKING = 1
    LOST = 2


class BallTracker:
    """
    Enhanced ball position tracker with prediction, spatial validation, and quality metrics
    """
    
    def __init__(self, max_history=50, fps=30, loss_threshold=5):
        """
        Initialize tracker
        
        Args:
            max_history: Maximum number of positions to store
            fps: Video frames per second (for velocity calculations)
            loss_threshold: Frames to consider ball lost
        """
        self.positions = deque(maxlen=max_history)
        self.timestamps = deque(maxlen=max_history)
        self.confidences = deque(maxlen=max_history)
        self.fps = fps
        self.loss_threshold = loss_threshold
        self.state = TrackingState.IDLE
        self.frames_lost = 0
        self.max_distance_pixels = 100  # Ball can't move more than this per frame
    
    def predict_position(self):
        """
        Predict next ball position using linear extrapolation
        
        Returns:
            Predicted position (x, y) or None
        """
        if len(self.positions) < 2:
            return None
        
        try:
            p1 = np.array(self.positions[-2], dtype=float)
            p2 = np.array(self.positions[-1], dtype=float)
            
            # Velocity vector
            velocity = p2 - p1
            
            # Predict next position
            predicted = p2 + velocity
            return tuple(map(int, predicted))
        except Exception as e:
            print(f"Prediction error: {e}")
            return None
    
    def update(self, detection, timestamp, validate_distance=True):
        """
        Update tracker with new detection or prediction
        
        Args:
            detection: Detection dict from BallDetector or None
            timestamp: Current timestamp (seconds)
            validate_distance: Enable spatial validation
            
        Returns:
            bool: True if successfully updated, False otherwise
        """
        if detection is not None:
            # Validate detection distance from last position
            if validate_distance and self.positions:
                last_pos = np.array(self.positions[-1], dtype=float)
                curr_pos = np.array(detection['position'], dtype=float)
                distance = np.linalg.norm(curr_pos - last_pos)
                
                if distance > self.max_distance_pixels:
                    print(f"Warning: Outlier detection ignored (distance: {distance:.1f}px)")
                    # Still predict to maintain tracking
                    return self._predict_and_update(timestamp)
            
            # Valid detection
            self.positions.append(detection['position'])
            self.timestamps.append(timestamp)
            self.confidences.append(detection['confidence'])
            self.frames_lost = 0
            self.state = TrackingState.TRACKING
            return True
        
        else:
            # No detection: use prediction to maintain track
            self.frames_lost += 1
            
            if self.frames_lost > self.loss_threshold:
                self.state = TrackingState.LOST
                return False
            
            if len(self.positions) >= 2:
                return self._predict_and_update(timestamp)
            
            return False
    
    def _predict_and_update(self, timestamp):
        """Helper to predict and update tracker"""
        predicted = self.predict_position()
        if predicted is not None:
            self.positions.append(predicted)
            self.timestamps.append(timestamp)
            self.confidences.append(0.0)  # Mark as predicted
            self.state = TrackingState.TRACKING
            return True
        return False
    
    def get_trajectory(self):
        """
        Get trajectory as list of positions
        
        Returns:
            list of (x, y) tuples
        """
        return list(self.positions)
    
    def get_trajectory_array(self):
        """
        Get trajectory as numpy array
        
        Returns:
            numpy array of shape (N, 2) with (x, y) positions
        """
        if not self.positions:
            return np.array([], dtype=np.float32).reshape(0, 2)
        return np.array(list(self.positions), dtype=np.float32)
    
    def get_velocity(self, window=3):
        """
        Calculate smoothed velocity using moving window average
        
        Args:
            window: Number of points to use for calculation
            
        Returns:
            Velocity in pixels/second
        """
        if len(self.positions) < window:
            return 0.0
        
        try:
            # Use last 'window' positions
            positions = np.array(list(self.positions)[-window:], dtype=float)
            timestamps = list(self.timestamps)[-window:]
            
            # Time span
            total_time = timestamps[-1] - timestamps[0]
            if total_time <= 0:
                return 0.0
            
            # Distance traveled
            total_distance = np.linalg.norm(positions[-1] - positions[0])
            
            return total_distance / total_time
        except Exception as e:
            print(f"Velocity calculation error: {e}")
            return 0.0
    
    def get_velocity_vector(self):
        """
        Get velocity as (vx, vy) component
        
        Returns:
            Tuple (velocity_x, velocity_y) in pixels/second
        """
        if len(self.positions) < 2:
            return (0.0, 0.0)
        
        try:
            p1 = np.array(self.positions[-2], dtype=float)
            p2 = np.array(self.positions[-1], dtype=float)
            dt = self.timestamps[-1] - self.timestamps[-2]
            
            if dt == 0:
                return (0.0, 0.0)
            
            velocity = (p2 - p1) / dt
            return tuple(velocity)
        except Exception as e:
            print(f"Velocity vector error: {e}")
            return (0.0, 0.0)
    
    def get_velocity_kmh(self, pixel_to_mm=0.5):
        """
        Convert velocity to km/h
        
        Args:
            pixel_to_mm: Calibration factor (1 pixel = X mm, adjust based on camera setup)
        
        Returns:
            Velocity in km/h
        """
        velocity_px_sec = self.get_velocity()
        velocity_mm_sec = velocity_px_sec * pixel_to_mm
        velocity_kmh = (velocity_mm_sec / 1000) * 3.6
        return velocity_kmh
    
    def get_trajectory_quality(self):
        """
        Calculate trajectory quality metric (0-1)
        Considers: confidence, continuity, and tracking state
        
        Returns:
            Quality score (0-1)
        """
        if len(self.confidences) == 0:
            return 0.0
        
        try:
            # Average detection confidence (exclude predicted frames)
            detection_confidences = [c for c in self.confidences if c > 0.0]
            if detection_confidences:
                avg_confidence = np.mean(detection_confidences)
            else:
                avg_confidence = 0.0
            
            # Penalize predicted (low confidence) frames
            predicted_frames = sum(1 for c in self.confidences if c == 0.0)
            gap_penalty = predicted_frames / max(len(self.confidences), 1)
            
            # Quality score
            quality = avg_confidence * (1 - gap_penalty * 0.3)
            
            return max(0.0, min(1.0, quality))
        except Exception as e:
            print(f"Quality calculation error: {e}")
            return 0.0
    
    def is_tracking(self):
        """Check if actively tracking"""
        return len(self.positions) > 0 and self.state != TrackingState.LOST
    
    def get_state(self):
        """Get current tracking state"""
        return self.state
    
    def get_frames_lost(self):
        """Get number of consecutive frames without detection"""
        return self.frames_lost
    
    def get_trajectory_length(self):
        """Get number of tracked positions"""
        return len(self.positions)
    
    def get_tracking_stats(self):
        """
        Get comprehensive tracking statistics
        
        Returns:
            dict with tracking metrics
        """
        return {
            'trajectory_length': self.get_trajectory_length(),
            'velocity_px_s': self.get_velocity(),
            'velocity_kmh': self.get_velocity_kmh(),
            'velocity_vector': self.get_velocity_vector(),
            'quality': self.get_trajectory_quality(),
            'state': self.get_state().name,
            'frames_lost': self.get_frames_lost(),
            'is_tracking': self.is_tracking()
        }
    
    def reset(self):
        """Reset tracker for new ball/delivery"""
        self.positions.clear()
        self.timestamps.clear()
        self.confidences.clear()
        self.state = TrackingState.IDLE
        self.frames_lost = 0