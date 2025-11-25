"""
Video Overlay Processor
Processes a video, detects sponsors using the trained model, 
and creates a new video with bounding boxes burned in.
"""

import cv2
import os
from pathlib import Path
from ultralytics import YOLO
import time

def process_video_with_overlay(input_path, output_path, model_path, conf_threshold=0.25):
    print("=" * 60)
    print("🎥 VIDEO PROCESSING WITH OVERLAY")
    print("=" * 60)
    
    # Check paths
    if not os.path.exists(input_path):
        print(f"❌ Input video not found: {input_path}")
        return False
        
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return False

    # Load model
    print(f"📦 Loading model from: {model_path}")
    
    # WORKAROUND: Copy model to a safe path (no apostrophes) to avoid Ultralytics bug
    import shutil
    safe_model_path = "temp_model_safe.pt"
    try:
        shutil.copy2(model_path, safe_model_path)
        print(f"   ↳ Copied to safe path: {safe_model_path}")
        model = YOLO(safe_model_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("❌ Error opening video file")
        return False

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📊 Video Info: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")

    # Setup output video writer
    # Try mp4v codec first, fallback to avc1 if needed
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"🚀 Processing video... (Output: {output_path})")
    
    start_time = time.time()
    frame_count = 0
    detections_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run detection
        results = model(frame, conf=conf_threshold, verbose=False)
        
        # Draw annotations
        annotated_frame = results[0].plot()
        
        # Count detections
        if results[0].boxes is not None:
            detections_count += len(results[0].boxes)

        # Write frame
        out.write(annotated_frame)
        
        frame_count += 1
        if frame_count % 50 == 0:
            elapsed = time.time() - start_time
            fps_proc = frame_count / elapsed
            progress = (frame_count / total_frames) * 100
            print(f"   Processed {frame_count}/{total_frames} frames ({progress:.1f}%) - {fps_proc:.1f} FPS")

    # Cleanup
    cap.release()
    out.release()
    
    total_time = time.time() - start_time
    print("-" * 60)
    print(f"✅ Processing complete!")
    print(f"⏱️ Time taken: {total_time:.1f}s")
    print(f"🎯 Total detections: {detections_count}")
    print(f"💾 Saved to: {output_path}")
    
    return True

if __name__ == "__main__":
    # Configuration
    INPUT_VIDEO = "video.mp4"
    OUTPUT_VIDEO = "video_output.mp4"
    
    # Find best model (using Path to avoid Windows issues)
    model_candidates = [
        Path("runs/detect/sponsor_detector/weights/best.pt"),
        Path("runs/detect/sponsor_detector2/weights/best.pt"),
        Path("runs/detect/sponsor_detector_v3/weights/best.pt")
    ]
    
    MODEL_PATH = None
    # Sort by modification time to get the latest
    existing_models = [p for p in model_candidates if p.exists()]
    if existing_models:
        MODEL_PATH = str(max(existing_models, key=os.path.getmtime).absolute())
    
    if MODEL_PATH:
        process_video_with_overlay(INPUT_VIDEO, OUTPUT_VIDEO, MODEL_PATH, conf_threshold=0.15)
    else:
        print("❌ No trained model found. Please train the model first.")
