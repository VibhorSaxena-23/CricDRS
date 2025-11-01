"""
Video processing utilities
"""

import cv2

class VideoProcessor:
    def __init__(self, input_path, output_path):
        """
        Initialize video processor
        
        Args:
            input_path: Input video file path
            output_path: Output video file path
        """
        self.cap = cv2.VideoCapture(input_path)
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_path, fourcc, self.fps, 
                                   (self.width, self.height))
        
    def read_frames(self):
        """Generator to read frames"""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame
    
    def write_frame(self, frame):
        """Write frame to output video"""
        self.out.write(frame)
    
    def release(self):
        """Release video resources"""
        self.cap.release()
        self.out.release()
