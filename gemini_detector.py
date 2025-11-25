"""
Gemini-based Sponsor Detection
Uses Google's Gemini 2.0 Flash model for sponsor logo detection
"""

import google.generativeai as genai
import cv2
import os
from pathlib import Path
from typing import List, Dict, Tuple
import json
import base64
from PIL import Image
import io

class GeminiSponsorDetector:
    """Detect sponsors using Gemini 2.0 Flash vision model"""
    
    def __init__(self, api_key: str):
        """Initialize Gemini detector"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Known sponsors to look for
        self.target_sponsors = [
            'Aramco', 'DP World', 'Emirates', 'Google', 
            'Rexona', 'Royal Stag', 'Pepsi', 'Coca Cola',
            'MRF', 'Jio', 'Dream11', 'CRED', 'PayTM'
        ]
    
    def detect_in_frame(self, frame_path: str, timestamp: float) -> List[Dict]:
        """
        Detect sponsors in a single frame using Gemini
        
        Args:
            frame_path: Path to frame image
            timestamp: Video timestamp
            
        Returns:
            List of detections with sponsor names and confidence
        """
        
        # Read image
        img = Image.open(frame_path)
        
        # Create prompt
        prompt = f"""Analyze this cricket match frame and identify any visible sponsor logos or brand names.

Look for these sponsors: {', '.join(self.target_sponsors)}

For each sponsor you find, provide:
1. Sponsor name (exact match from the list above)
2. Confidence level (high/medium/low)
3. Location description (e.g., "jersey", "boundary board", "LED screen", "stumps")

Respond ONLY with a JSON array in this exact format:
[
  {{"sponsor": "Aramco", "confidence": "high", "location": "boundary board"}},
  {{"sponsor": "Emirates", "confidence": "medium", "location": "jersey"}}
]

If no sponsors are visible, return an empty array: []

Do not include any other text, only the JSON array."""

        try:
            # Generate response
            response = self.model.generate_content([prompt, img])
            
            # Parse JSON response
            response_text = response.text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
            
            detections_raw = json.loads(response_text)
            
            # Format detections
            detections = []
            for det in detections_raw:
                # Map confidence to numeric value
                conf_map = {'high': 0.9, 'medium': 0.6, 'low': 0.3}
                confidence = conf_map.get(det.get('confidence', 'low').lower(), 0.5)
                
                detections.append({
                    'sponsor': det['sponsor'],
                    'confidence': confidence,
                    'location': det.get('location', 'unknown'),
                    'timestamp': timestamp,
                    'frame_path': frame_path
                })
            
            return detections
            
        except Exception as e:
            print(f"⚠️ Error detecting in frame {frame_path}: {e}")
            return []
    
    def process_video_quick(self, video_path: str, sample_interval: int = 30) -> Tuple[List[Dict], str]:
        """
        Process video by sampling frames
        
        Args:
            video_path: Path to video
            sample_interval: Sample every Nth frame
            
        Returns:
            Tuple of (detections, preview_dir)
        """
        
        # Create preview directory
        preview_dir = "preview_frames"
        os.makedirs(preview_dir, exist_ok=True)
        
        # Clean old previews
        for f in Path(preview_dir).glob("*"):
            f.unlink()
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Cannot open video")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        all_detections = []
        frame_count = 0
        processed_count = 0
        
        print(f"📹 Processing video: {total_frames} frames @ {fps} FPS")
        print(f"🎯 Sampling every {sample_interval} frames with Gemini 2.0 Flash")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frame
            if frame_count % sample_interval == 0:
                timestamp = frame_count / fps
                
                # Save frame temporarily
                temp_frame = os.path.join(preview_dir, f"temp_frame_{processed_count}.jpg")
                cv2.imwrite(temp_frame, frame)
                
                # Detect with Gemini
                print(f"🔍 Analyzing frame {frame_count} ({timestamp:.1f}s)...")
                detections = self.detect_in_frame(temp_frame, timestamp)
                
                if detections:
                    # Draw detections on frame
                    annotated_frame = self.draw_detections(frame, detections)
                    
                    # Save annotated frame
                    final_frame = os.path.join(preview_dir, f"frame_{processed_count:04d}_t{int(timestamp)}s.jpg")
                    cv2.imwrite(final_frame, annotated_frame)
                    
                    # Update frame paths
                    for det in detections:
                        det['frame_path'] = final_frame
                    
                    all_detections.extend(detections)
                    print(f"   ✅ Found {len(detections)} sponsor(s): {', '.join(d['sponsor'] for d in detections)}")
                    
                    # Remove temp file
                    os.remove(temp_frame)
                else:
                    print(f"   ⚠️ No sponsors detected")
                    os.remove(temp_frame)
                
                processed_count += 1
            
            frame_count += 1
            
            # Progress
            if frame_count % 500 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"   Progress: {progress:.1f}%")
        
        cap.release()
        
        print(f"\n📊 Summary:")
        print(f"   Frames analyzed: {processed_count}")
        print(f"   Total detections: {len(all_detections)}")
        
        return all_detections, preview_dir
    
    def draw_detections(self, frame, detections: List[Dict]):
        """Draw detection labels on frame (simple text overlay)"""
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        
        # Draw sponsor names at top
        y_offset = 30
        for det in detections:
            text = f"{det['sponsor']} ({det['location']}) - {det['confidence']:.0%}"
            cv2.putText(annotated, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
        
        return annotated

if __name__ == "__main__":
    # Test
    API_KEY = "AIzaSyBJkNDE85DZRj6xy9rxi61QZcefNnkDFBA"
    detector = GeminiSponsorDetector(API_KEY)
    
    detections, preview_dir = detector.process_video_quick("video.mp4", sample_interval=30)
    print(f"\n✅ Preview generated in: {preview_dir}/")
