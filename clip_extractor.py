from moviepy.editor import VideoFileClip
import os
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class ClipExtractor:
    """Extract video clips based on brand detections"""
    
    def __init__(self, output_dir: str = "brand_clips"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def extract_brand_clips(self, video_path: str, detections: List[Dict], 
                           clip_duration: float = 5.0) -> Dict[str, List[str]]:
        """
        Extract clips for each brand detection
        
        Args:
            video_path: Path to original video
            detections: List of detection dictionaries
            clip_duration: Duration of each clip in seconds
            
        Returns:
            Dictionary mapping brand names to clip file paths
        """
        if not detections:
            return {}
            
        # Load video
        video = VideoFileClip(video_path)
        brand_clips = {}
        
        # Group detections by brand
        brand_detections = {}
        for det in detections:
            brand = det['brand_name']
            if brand not in brand_detections:
                brand_detections[brand] = []
            brand_detections[brand].append(det)
        
        # Extract clips for each brand
        for brand, brand_dets in brand_detections.items():
            brand_clips[brand] = []
            
            # Sort by timestamp
            brand_dets.sort(key=lambda x: x['timestamp'])
            
            # Extract clips (avoid overlapping clips)
            last_end_time = 0
            for i, det in enumerate(brand_dets):
                start_time = max(det['timestamp'] - clip_duration/2, 0, last_end_time)
                end_time = min(start_time + clip_duration, video.duration)
                
                if end_time - start_time >= 2.0:  # Minimum 2 second clips
                    clip_filename = f"{brand}_clip_{i+1:03d}_{start_time:.1f}s.mp4"
                    clip_path = os.path.join(self.output_dir, clip_filename)
                    
                    # Extract clip
                    clip = video.subclip(start_time, end_time)
                    clip.write_videofile(clip_path, verbose=False, logger=None)
                    
                    brand_clips[brand].append(clip_path)
                    last_end_time = end_time
        
        video.close()
        logger.info(f"Extracted clips for {len(brand_clips)} brands")
        return brand_clips
