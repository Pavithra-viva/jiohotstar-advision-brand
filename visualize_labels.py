"""
Visualize YOLO labels on images to verify labeling quality
"""

import cv2
import os
from pathlib import Path

def visualize_labels():
    """Draw bounding boxes from label files on images"""
    
    images_dir = Path("datasets/datasets")
    labels_dir = Path("datasets/labels")
    output_dir = Path("label_visualization")
    output_dir.mkdir(exist_ok=True)
    
    # Class names
    class_names = {
        0: 'aramco',
        1: 'dpworld',
        2: 'emirates',
        3: 'google',
        4: 'rexona',
        5: 'royalstag'
    }
    
    # Colors for each class
    colors = {
        0: (255, 0, 0),      # Blue
        1: (0, 255, 0),      # Green
        2: (0, 0, 255),      # Red
        3: (255, 255, 0),    # Cyan
        4: (255, 0, 255),    # Magenta
        5: (0, 255, 255)     # Yellow
    }
    
    print("=" * 60)
    print("🖼️ VISUALIZING YOLO LABELS")
    print("=" * 60)
    
    # Get all images
    image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
    
    total_boxes = 0
    images_with_labels = 0
    
    for img_path in image_files:
        label_path = labels_dir / f"{img_path.stem}.txt"
        
        if not label_path.exists():
            print(f"⚠️ No label: {img_path.name}")
            continue
        
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        h, w = img.shape[:2]
        
        # Read labels
        with open(label_path, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        
        if not lines:
            print(f"⚠️ Empty label: {img_path.name}")
            continue
        
        images_with_labels += 1
        box_count = 0
        
        # Draw each bounding box
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            # Convert YOLO format to pixel coordinates
            x1 = int((x_center - width/2) * w)
            y1 = int((y_center - height/2) * h)
            x2 = int((x_center + width/2) * w)
            y2 = int((y_center + height/2) * h)
            
            # Draw box
            color = colors.get(class_id, (128, 128, 128))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = class_names.get(class_id, f"class_{class_id}")
            label_text = f"{label} ({width*w:.0f}x{height*h:.0f}px)"
            
            # Background for text
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
            cv2.putText(img, label_text, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            box_count += 1
            total_boxes += 1
            
            # Warn about tiny boxes
            if width < 0.01 or height < 0.01:
                print(f"   ⚠️ Tiny box in {img_path.name}: {width:.4f}x{height:.4f}")
        
        # Save visualization
        output_path = output_dir / f"labeled_{img_path.name}"
        cv2.imwrite(str(output_path), img)
        
        print(f"✅ {img_path.name}: {box_count} box(es) → {output_path.name}")
    
    print("\n" + "=" * 60)
    print(f"📊 SUMMARY")
    print("=" * 60)
    print(f"Total images: {len(image_files)}")
    print(f"Images with labels: {images_with_labels}")
    print(f"Total bounding boxes: {total_boxes}")
    print(f"\n💾 Visualizations saved to: {output_dir}/")
    print("\n📌 Check the visualizations to verify labels are correct!")

if __name__ == "__main__":
    visualize_labels()
