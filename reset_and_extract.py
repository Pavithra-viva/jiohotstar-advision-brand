"""
Reset Dataset and Extract New Frames
1. Deletes old dataset images and labels.
2. Extracts fresh frames from video.mp4.
"""

import shutil
import os
from pathlib import Path
import cv2

def reset_and_extract():
    print("=" * 60)
    print("🔄 RESETTING DATASET & EXTRACTING FRAMES")
    print("=" * 60)

    # 1. Clean Directories
    dirs_to_clean = [
        Path("datasets/datasets"),
        Path("datasets/labels"),
        Path("datasets/images"),
        Path("datasets/labels_yolo"),
        Path("datasets/new_training_data")
    ]

    print("🧹 Cleaning old data...")
    for d in dirs_to_clean:
        if d.exists():
            try:
                shutil.rmtree(d)
                print(f"   ✅ Removed {d}")
            except Exception as e:
                print(f"   ⚠️ Could not remove {d}: {e}")
    
    # Re-create necessary dirs
    Path("datasets/datasets").mkdir(parents=True, exist_ok=True)
    Path("datasets/labels").mkdir(parents=True, exist_ok=True)

    # 2. Extract Frames
    video_path = "video.mp4"
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return

    print(f"\n🎬 Extracting frames from {video_path}...")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("❌ Error opening video file")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"   Duration: {duration:.1f}s, FPS: {fps:.1f}")

    # Extract ~60 frames evenly spaced
    target_frames = 60
    interval = max(1, int(total_frames / target_frames))
    
    count = 0
    saved = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if count % interval == 0:
            # Save frame
            frame_name = f"frame_{saved:04d}.jpg"
            save_path = Path("datasets/datasets") / frame_name
            cv2.imwrite(str(save_path), frame)
            saved += 1
            if saved % 10 == 0:
                print(f"   Saved {saved} frames...")
        
        count += 1
        if saved >= target_frames:
            break
            
    cap.release()
    print(f"\n✅ Extraction complete! {saved} frames saved to datasets/datasets/")
    print("\n📌 Next Step: Run 'python relabel_tool.py' to label the new frames.")

if __name__ == "__main__":
    reset_and_extract()
