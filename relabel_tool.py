"""
Manual Re-labeling Tool
Workflow:
1. Show image
2. Draw box (drag mouse)
3. Press number (0-5) to assign class
4. Press 's' to save and next
"""

import cv2
import os
from pathlib import Path

# Configuration
IMAGES_DIR = Path("datasets/datasets")
LABELS_DIR = Path("datasets/labels")
LABELS_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = {
    0: 'aramco',
    1: 'dpworld',
    2: 'emirates',
    3: 'google',
    4: 'rexona',
    5: 'royalstag'
}

def relabel_images():
    print("=" * 60)
    print("🏷️ MANUAL RE-LABELING TOOL")
    print("=" * 60)
    print("Controls:")
    print("  Mouse Drag : Draw bounding box")
    print("  0-5        : Assign class ID")
    print("  s          : Save labels and go to next image")
    print("  c          : Clear current boxes")
    print("  n          : Skip image (keep existing labels if any)")
    print("  q          : Quit")
    print("-" * 60)

    image_files = list(IMAGES_DIR.glob("*.png")) + list(IMAGES_DIR.glob("*.jpg"))
    
    if not image_files:
        print("❌ No images found in datasets/datasets/")
        return

    cv2.namedWindow("Labeling", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Labeling", 1200, 800)

    for i, img_path in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}] Processing: {img_path.name}")
        
        img = cv2.imread(str(img_path))
        if img is None:
            print("  ❌ Could not read image.")
            continue

        # State
        boxes = []
        current_box = None
        drawing = False
        ix, iy = -1, -1

        def mouse_callback(event, x, y, flags, param):
            nonlocal ix, iy, drawing, current_box
            
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                ix, iy = x, y
                current_box = (x, y, x, y)

            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    current_box = (ix, iy, x, y)

            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                current_box = (ix, iy, x, y)
                # Normalize coordinates (x1, y1, x2, y2)
                x1, y1, x2, y2 = current_box
                boxes.append({'coords': (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)), 'class_id': -1})
                current_box = None

        cv2.setMouseCallback("Labeling", mouse_callback)

        while True:
            display_img = img.copy()

            # Draw saved boxes
            for box in boxes:
                x1, y1, x2, y2 = box['coords']
                cid = box['class_id']
                
                color = (0, 255, 0) if cid != -1 else (0, 0, 255) # Green if classified, Red if not
                cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)
                
                label = CLASSES.get(cid, "?") if cid != -1 else "?"
                cv2.putText(display_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw box currently being drawn
            if current_box:
                x1, y1, x2, y2 = current_box
                cv2.rectangle(display_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            cv2.imshow("Labeling", display_img)
            key = cv2.waitKey(1) & 0xFF

            # Class assignment
            if ord('0') <= key <= ord('5'):
                cid = key - ord('0')
                if boxes:
                    # Assign to the last drawn box that doesn't have a class yet, or just the last one
                    # Logic: User draws box, then immediately presses number. So apply to last box.
                    boxes[-1]['class_id'] = cid
                    print(f"  ✅ Assigned class {cid} ({CLASSES[cid]}) to last box.")
            
            # Save and Next
            elif key == ord('s'):
                # Validate all boxes have classes
                if any(b['class_id'] == -1 for b in boxes):
                    print("  ⚠️ Warning: Some boxes have no class assigned (marked red). Assign class (0-5) before saving.")
                    continue
                
                save_labels(img_path, boxes, img.shape)
                print(f"  💾 Saved {len(boxes)} labels.")
                break

            # Clear
            elif key == ord('c'):
                boxes = []
                print("  Cleared boxes.")

            # Skip
            elif key == ord('n'):
                print("  Skipped.")
                break

            # Quit
            elif key == ord('q'):
                print("Exiting...")
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print("\n✅ Re-labeling complete!")

def save_labels(img_path, boxes, shape):
    h, w = shape[:2]
    label_path = LABELS_DIR / f"{img_path.stem}.txt"
    
    with open(label_path, 'w') as f:
        for box in boxes:
            x1, y1, x2, y2 = box['coords']
            cid = box['class_id']
            
            # Convert to YOLO format
            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h
            width = (x2 - x1) / w
            height = (y2 - y1) / h
            
            f.write(f"{cid} {x_center} {y_center} {width} {height}\n")

if __name__ == "__main__":
    relabel_images()
