import cv2
import os
import numpy as np
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoProcessor:
    """Handles video processing and frame extraction"""
    
    def __init__(self, frame_interval: int = 30):
        """
        Initialize VideoProcessor
        
        Args:
            frame_interval: Extract every nth frame (default: 30 = ~1 frame per second for 30fps video)
        """
        self.frame_interval = frame_interval
        self.frames_dir = "extracted_frames"
        
    def extract_frames(self, video_path: str) -> Tuple[List[str], dict]:
        """
        Extract frames from video at specified intervals
        
        Args:
            video_path: Path to input video file
            
        Returns:
            Tuple of (frame_paths, video_info)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        # Create frames directory
        os.makedirs(self.frames_dir, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        video_info = {
            'fps': fps,
            'total_frames': total_frames,
            'duration': duration,
            'width': width,
            'height': height,
            'filename': os.path.basename(video_path)
        }
        
        logger.info(f"Processing video: {video_info['filename']}")
        logger.info(f"Duration: {duration:.2f}s, FPS: {fps}, Resolution: {width}x{height}")
        
        frame_paths = []
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Extract frame at specified interval
            if frame_count % self.frame_interval == 0:
                timestamp = frame_count / fps
                frame_filename = f"frame_{extracted_count:06d}_t{timestamp:.2f}s.jpg"
                frame_path = os.path.join(self.frames_dir, frame_filename)
                
                # Save frame
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
                extracted_count += 1
                
                if extracted_count % 100 == 0:
                    logger.info(f"Extracted {extracted_count} frames...")
                    
            frame_count += 1
            
        cap.release()
        logger.info(f"Extraction complete: {extracted_count} frames extracted")
        
        return frame_paths, video_info
    
    def get_frame_timestamp(self, frame_path: str) -> float:
        """Extract timestamp from frame filename"""
        try:
            # Extract timestamp from filename like "frame_000001_t12.34s.jpg"
            filename = os.path.basename(frame_path)
            timestamp_part = filename.split('_t')[1].split('s.jpg')[0]
            return float(timestamp_part)
        except:
            return 0.0
    
    def cleanup_frames(self):
        """Remove extracted frames directory"""
        import shutil
        if os.path.exists(self.frames_dir):
            shutil.rmtree(self.frames_dir)
            logger.info("Cleaned up extracted frames")
