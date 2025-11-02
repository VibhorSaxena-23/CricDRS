"""
Enhanced Ball Detection using YOLOv8 with filtering and validation
"""
from ultralytics import YOLO
import torch
import numpy as np
from collections import deque
import time


class BallDetector:
    """
    Enhanced ball detection using YOLOv8 with spatial and size validation
    """
    
    def __init__(self, model_path, confidence=0.5, device='cpu'):
        """
        Initialize ball detector
        
        Args:
            model_path: Path to YOLOv8 model
            confidence: Detection confidence threshold (0-1)
            device: 'cpu' or 'cuda'
        """
        try:
            self.model = YOLO(model_path)
            self.confidence = confidence
            # Use GPU if available, fallback to CPU
            self.device = device if torch.cuda.is_available() else 'cpu'
            self.model.to(self.device)
            self.last_position = None
            self.inference_times = deque(maxlen=30)  # Track last 30 inferences
            
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")
    
    def detect(self, frame):
        """
        Detect ball in frame with spatial and size validation
        
        Args:
            frame: Input image (numpy array, BGR format)
            
        Returns:
            dict with 'position', 'bbox', 'confidence', 'width', 'height' or None
        """
        if frame is None or frame.size == 0:
            return None
        
        try:
            start_time = time.time()
            
            # Run YOLO inference
            results = self.model(frame, conf=self.confidence, verbose=False)
            boxes = results[0].boxes
            
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            if len(boxes) == 0:
                return None
            
            # Filter detections by cricket ball size constraints
            # Cricket ball: ~7cm diameter
            # Expected pixel range: 15-150px depending on camera angle
            valid_detections = []
            
            for i, box in enumerate(boxes):
                x, y, w, h = box.xywh[0].cpu().numpy()
                
                # Size validation
                if 15 < w < 150 and 15 < h < 150:
                    # Aspect ratio check (ball should be roughly circular)
                    aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
                    if 0.7 < aspect_ratio < 1.4:
                        valid_detections.append({
                            'index': i,
                            'box': box,
                            'x': x,
                            'y': y,
                            'w': w,
                            'h': h,
                            'confidence': float(box.conf[0].cpu().numpy())
                        })
            
            if not valid_detections:
                return None
            
            # Select best detection: prioritize temporal coherence with last position
            if self.last_position and len(valid_detections) > 1:
                # Prefer detection closest to previous position
                best_det = min(
                    valid_detections,
                    key=lambda d: np.linalg.norm(
                        np.array([d['x'], d['y']]) - np.array(self.last_position)
                    ) - d['confidence'] * 50  # Boost confidence score
                )
            else:
                # Highest confidence detection
                best_det = max(valid_detections, key=lambda d: d['confidence'])
            
            position = (int(best_det['x']), int(best_det['y']))
            self.last_position = position
            
            return {
                'position': position,
                'bbox': (
                    int(best_det['x'] - best_det['w']/2),
                    int(best_det['y'] - best_det['h']/2),
                    int(best_det['w']),
                    int(best_det['h'])
                ),
                'confidence': best_det['confidence'],
                'width': int(best_det['w']),
                'height': int(best_det['h']),
                'inference_time': inference_time
            }
        
        except RuntimeError as e:
            print(f"YOLO inference error: {e}")
            return None
        except Exception as e:
            print(f"Detection error: {e}")
            return None
    
    def get_avg_inference_time(self):
        """Get average inference time (ms)"""
        if not self.inference_times:
            return 0.0
        return np.mean(self.inference_times) * 1000
    
    def reset(self):
        """Reset detector state"""
        self.last_position = None