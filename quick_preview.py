"""
Quick Preview Generator
Uses Gemini 2.0 Flash for sponsor detection (client-facing, AI hidden)
"""

from gemini_detector import GeminiSponsorDetector
from typing import List, Tuple, Dict
import streamlit as st

def get_api_key():
    """Get Gemini API key from secrets or fallback"""
    try:
        return st.secrets["gemini"]["api_key"]
    except:
        # Fallback for testing outside Streamlit
        return "AIzaSyBJkNDE85DZRj6xy9rxi61QZcefNnkDFBA"

def generate_quick_preview(video_path: str, sample_interval: int = 30, conf_threshold: float = 0.15) -> Tuple[List[Dict], str]:
    """
    Generate quick preview by sampling frames
    
    Args:
        video_path: Path to input video
        sample_interval: Sample every Nth frame
        conf_threshold: Minimum confidence threshold (filters Gemini results)
        
    Returns:
        Tuple of (detections_list, preview_dir)
    """
    
    print(f"🚀 Starting sponsor detection analysis...")
    
    # Initialize Gemini detector
    detector = GeminiSponsorDetector(get_api_key())
    
    # Process video
    all_detections, preview_dir = detector.process_video_quick(video_path, sample_interval)
    
    # Filter by confidence threshold
    filtered_detections = [d for d in all_detections if d['confidence'] >= conf_threshold]
    
    print(f"\n✅ Analysis complete!")
    print(f"   Total detections: {len(all_detections)}")
    print(f"   After filtering (conf >= {conf_threshold}): {len(filtered_detections)}")
    
    return filtered_detections, preview_dir

if __name__ == "__main__":
    # Test
    detections, preview_dir = generate_quick_preview("video.mp4", sample_interval=30, conf_threshold=0.3)
    print(f"\n✅ Preview generated in: {preview_dir}/")
