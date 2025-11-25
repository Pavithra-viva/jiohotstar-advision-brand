"""
Video Frame Extraction and Auto-Labeling Tool
Extracts frames from cricket videos and helps label sponsor logos
"""

import cv2
import os
from pathlib import Path
import numpy as np
from ultralytics import YOLO
import shutil

class VideoFrameExtractor:
    def __init__(self, video_path, output_dir="datasets/new_training_data"):
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        
        # Create directories
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Load base YOLO model for initial detection
        self.detector = YOLO('yolov8n.pt')
        
        # Sponsor brand names (from your dataset)
        self.sponsor_classes = {
            0: 'aramco',
            1: 'dpworld',
            2: 'emirates',
            3: 'google',
            4: 'rexona',
            5: 'royalstag'
        }
        
    def extract_frames(self, interval_seconds=2, max_frames=50):
        """Extract frames from video at regular intervals"""
        print(f"🎬 Extracting frames from: {self.video_path}")
        
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print("❌ Error: Cannot open video file")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"📊 Video Info:")
        print(f"   Duration: {duration:.1f}s")
        print(f"   FPS: {fps:.1f}")
        print(f"   Total Frames: {total_frames}")
        
        # Calculate frame interval
        frame_interval = int(fps * interval_seconds)
        
        extracted_frames = []
        frame_count = 0
        saved_count = 0
        
        while cap.isOpened() and saved_count < max_frames:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Extract frame at intervals
            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps
                
                # Save frame
                frame_name = f"frame_{saved_count:04d}_t{int(timestamp):04d}s.jpg"
                frame_path = self.images_dir / frame_name
                
                cv2.imwrite(str(frame_path), frame)
                extracted_frames.append({
                    'path': frame_path,
                    'frame_number': frame_count,
                    'timestamp': timestamp
                })
                
                print(f"   ✅ Extracted frame {saved_count + 1}/{max_frames} at {timestamp:.1f}s")
                saved_count += 1
            
            frame_count += 1
        
        cap.release()
        
        print(f"\n✅ Extracted {len(extracted_frames)} frames")
        return extracted_frames
    
    def auto_detect_objects(self, frame_path):
        """Use base YOLO to detect potential sponsor locations"""
        results = self.detector(str(frame_path), conf=0.3, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.detector.names[class_id]
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class': class_name
                    })
        
        return detections
    
    def create_interactive_labeler(self, extracted_frames):
        """Interactive labeling tool with keyboard shortcuts"""
        print("\n" + "=" * 60)
        print("🏷️ INTERACTIVE LABELING TOOL")
        print("=" * 60)
        print("\nKeyboard Shortcuts:")
        print("  0-5: Assign sponsor class (0=aramco, 1=dpworld, 2=emirates, 3=google, 4=rexona, 5=royalstag)")
        print("  n: Next frame (skip current)")
        print("  s: Save and next")
        print("  q: Quit")
        print("\nMouse:")
        print("  Click and drag to draw bounding box around sponsor logo")
        print("=" * 60)
        
        current_idx = 0
        
        while current_idx < len(extracted_frames):
            frame_info = extracted_frames[current_idx]
            frame_path = frame_info['path']
            
            img = cv2.imread(str(frame_path))
            if img is None:
                current_idx += 1
                continue
            
            # Create window
            window_name = f"Label Frame {current_idx + 1}/{len(extracted_frames)} - {frame_path.name}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1200, 800)
            
            # Drawing state
            drawing = False
            ix, iy = -1, -1
            current_class = 0
            boxes = []
            
            def mouse_callback(event, x, y, flags, param):
                nonlocal drawing, ix, iy, boxes, img
                
                if event == cv2.EVENT_LBUTTONDOWN:
                    drawing = True
                    ix, iy = x, y
                
                elif event == cv2.EVENT_MOUSEMOVE and drawing:
                    img_copy = img.copy()
                    # Draw existing boxes
                    for box_info in boxes:
                        cv2.rectangle(img_copy, 
                                    (int(box_info['x1']), int(box_info['y1'])),
                                    (int(box_info['x2']), int(box_info['y2'])),
                                    (0, 255, 0), 2)
                        label = self.sponsor_classes[box_info['class']]
                        cv2.putText(img_copy, label, 
                                  (int(box_info['x1']), int(box_info['y1']) - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Draw current box
                    cv2.rectangle(img_copy, (ix, iy), (x, y), (255, 0, 0), 2)
                    cv2.imshow(window_name, img_copy)
                
                elif event == cv2.EVENT_LBUTTONUP:
                    drawing = False
                    boxes.append({
                        'x1': min(ix, x),
                        'y1': min(iy, y),
                        'x2': max(ix, x),
                        'y2': max(iy, y),
                        'class': current_class
                    })
                    
                    # Redraw with all boxes
                    img_copy = img.copy()
                    for box_info in boxes:
                        cv2.rectangle(img_copy, 
                                    (int(box_info['x1']), int(box_info['y1'])),
                                    (int(box_info['x2']), int(box_info['y2'])),
                                    (0, 255, 0), 2)
                        label = self.sponsor_classes[box_info['class']]
                        cv2.putText(img_copy, label, 
                                  (int(box_info['x1']), int(box_info['y1']) - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.imshow(window_name, img_copy)
            
            cv2.setMouseCallback(window_name, mouse_callback)
            cv2.imshow(window_name, img)
            
            # Show instructions
            print(f"\n📸 Frame {current_idx + 1}/{len(extracted_frames)}: {frame_path.name}")
            print(f"   Current class: {current_class} ({self.sponsor_classes[current_class]})")
            
            while True:
                key = cv2.waitKey(1) & 0xFF
                
                # Class selection (0-5)
                if key >= ord('0') and key <= ord('5'):
                    current_class = key - ord('0')
                    print(f"   Selected class: {current_class} ({self.sponsor_classes[current_class]})")
                
                # Save and next
                elif key == ord('s'):
                    if boxes:
                        self.save_labels(frame_path, boxes, img.shape)
                        print(f"   ✅ Saved {len(boxes)} label(s)")
                    current_idx += 1
                    break
                
                # Skip
                elif key == ord('n'):
                    print("   ⏭️ Skipped")
                    current_idx += 1
                    break
                
                # Quit
                elif key == ord('q'):
                    cv2.destroyAllWindows()
                    return
            
            cv2.destroyAllWindows()
        
        print("\n✅ Labeling complete!")
    
    def save_labels(self, image_path, boxes, img_shape):
        """Save labels in YOLO format"""
        h, w = img_shape[:2]
        label_path = self.labels_dir / f"{image_path.stem}.txt"
        
        with open(label_path, 'w') as f:
            for box in boxes:
                # Convert to YOLO format (normalized center x, y, width, height)
                x_center = ((box['x1'] + box['x2']) / 2) / w
                y_center = ((box['y1'] + box['y2']) / 2) / h
                width = (box['x2'] - box['x1']) / w
                height = (box['y2'] - box['y1']) / h
                
                f.write(f"{box['class']} {x_center} {y_center} {width} {height}\n")
    
    def merge_with_existing_dataset(self):
        """Merge new labeled data with existing dataset"""
        print("\n🔄 Merging with existing dataset...")
        
        existing_images = Path("datasets/datasets")
        existing_labels = Path("datasets/labels")
        
        # Copy new images
        for img in self.images_dir.glob("*.jpg"):
            label_file = self.labels_dir / f"{img.stem}.txt"
            if label_file.exists():
                shutil.copy(img, existing_images / img.name)
                shutil.copy(label_file, existing_labels / label_file.name)
                print(f"   ✅ Added {img.name}")
        
        print("✅ Dataset merged!")

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python extract_and_label.py <video_path>")
        print("\nExample:")
        print("  python extract_and_label.py cricket_match.mp4")
        return
    
    video_path = sys.argv[1]
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return
    
    print("=" * 60)
    print("🏏 VIDEO FRAME EXTRACTION & LABELING TOOL")
    print("=" * 60)
    
    extractor = VideoFrameExtractor(video_path)
    
    # Extract frames
    frames = extractor.extract_frames(interval_seconds=3, max_frames=30)
    
    if not frames:
        print("❌ No frames extracted")
        return
    
    # Interactive labeling
    extractor.create_interactive_labeler(frames)
    
    # Merge with existing dataset
    response = input("\n🔄 Merge with existing dataset? (y/n): ")
    if response.lower() == 'y':
        extractor.merge_with_existing_dataset()
        print("\n✅ Ready to retrain! Run: python train_sponsor_model.py")

if __name__ == "__main__":
    main()
