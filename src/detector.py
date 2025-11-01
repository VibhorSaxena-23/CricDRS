"""
Ball detection using YOLOv8
"""

from ultralytics import YOLO
import torch

class BallDetector:
    def __init__(self, model_path, confidence=0.5, device='cpu'):
        """
        Initialize ball detector
        
        Args:
            model_path: Path to YOLOv8 model
            confidence: Detection confidence threshold
            device: 'cpu' or 'cuda'
        """
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.device = device
        
        # Force to CPU/GPU
        self.model.to(device)
        
    def detect(self, frame):
        """
        Detect ball in frame
        
        Args:
            frame: Input image (numpy array)
            
        Returns:
            dict with 'position', 'bbox', 'confidence' or None
        """
        # Run inference
        results = self.model(frame, conf=self.confidence, verbose=False)
        
        # Extract detections
        boxes = results[0].boxes
        
        if len(boxes) > 0:
            # Get highest confidence detection
            best_idx = boxes.conf.argmax()
            box = boxes[best_idx]
            
            # Extract coordinates
            x, y, w, h = box.xywh[0].cpu().numpy()
            confidence = box.conf[0].cpu().numpy()
            
            return {
                'position': (int(x), int(y)),
                'bbox': (int(x - w/2), int(y - h/2), int(w), int(h)),
                'confidence': float(confidence)
            }
        
        return None
