"""
Diagnostic script to identify and fix detection issues
"""

import os
import re
from pathlib import Path
from ultralytics import YOLO
import cv2

def check_labels():
    """Check if labels match the expected class IDs"""
    print("=" * 60)
    print("🔍 DIAGNOSING LABEL FILES")
    print("=" * 60)
    
    labels_dir = Path("datasets/labels")
    
    # Expected class mapping
    expected_classes = {
        'aramco': 0,
        'dpworld': 1,
        'emirates': 2,
        'google': 3,
        'rexona': 4,
        'royalstag': 5
    }
    
    print("\n📋 Expected Class Mapping:")
    for name, id in expected_classes.items():
        print(f"   {id}: {name}")
    
    print("\n📄 Checking Label Files:")
    
    issues_found = []
    
    for label_file in labels_dir.glob("*.txt"):
        filename = label_file.stem
        
        # Extract class name from filename
        class_name = re.split(r'[_\d]', filename)[0].lower()
        expected_id = expected_classes.get(class_name, -1)
        
        # Read label file
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            if line.strip():
                parts = line.strip().split()
                actual_id = int(parts[0])
                
                if actual_id != expected_id:
                    issues_found.append({
                        'file': label_file.name,
                        'class_name': class_name,
                        'expected_id': expected_id,
                        'actual_id': actual_id
                    })
                    print(f"   ⚠️ {label_file.name}: class_id={actual_id} (expected {expected_id} for '{class_name}')")
                else:
                    print(f"   ✅ {label_file.name}: class_id={actual_id} ✓")
    
    return issues_found

def fix_labels():
    """Fix label class IDs to match filenames"""
    print("\n" + "=" * 60)
    print("🔧 FIXING LABEL FILES")
    print("=" * 60)
    
    labels_dir = Path("datasets/labels")
    
    class_mapping = {
        'aramco': 0,
        'dpworld': 1,
        'emirates': 2,
        'google': 3,
        'rexona': 4,
        'royalstag': 5
    }
    
    fixed_count = 0
    
    for label_file in labels_dir.glob("*.txt"):
        filename = label_file.stem
        class_name = re.split(r'[_\d]', filename)[0].lower()
        correct_id = class_mapping.get(class_name)
        
        if correct_id is None:
            print(f"   ⚠️ Unknown class: {class_name}")
            continue
        
        # Read and fix labels
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            if line.strip():
                parts = line.strip().split()
                # Replace class ID with correct one
                parts[0] = str(correct_id)
                new_lines.append(' '.join(parts) + '\n')
        
        # Write back
        with open(label_file, 'w') as f:
            f.writelines(new_lines)
        
        print(f"   ✅ Fixed {label_file.name} → class_id={correct_id}")
        fixed_count += 1
    
    print(f"\n✅ Fixed {fixed_count} label files!")

def test_detection_with_low_confidence():
    """Test model with very low confidence threshold"""
    print("\n" + "=" * 60)
    print("🧪 TESTING MODEL WITH LOW CONFIDENCE")
    print("=" * 60)
    
    model_path = "runs/detect/sponsor_detector/weights/best.pt"
    
    if not os.path.exists(model_path):
        print("❌ Model not found! Train the model first.")
        return
    
    model = YOLO(model_path)
    
    # Test on multiple images with very low confidence
    test_images = list(Path("datasets/datasets").glob("*.png"))[:3]
    
    for img_path in test_images:
        print(f"\n📸 Testing: {img_path.name}")
        
        # Try with very low confidence (0.01)
        results = model(str(img_path), conf=0.01, verbose=False)
        
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                print(f"   ✅ Found {len(boxes)} detection(s):")
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = model.names[class_id]
                    print(f"      - {class_name}: {confidence:.2%} confidence")
            else:
                print(f"   ❌ No detections (even at 1% confidence)")
        
        # Save annotated image
        annotated = results[0].plot()
        output_path = f"debug_{img_path.stem}.jpg"
        cv2.imwrite(output_path, annotated)
        print(f"   💾 Saved: {output_path}")

def check_training_images():
    """Verify training images exist and are readable"""
    print("\n" + "=" * 60)
    print("🖼️ CHECKING TRAINING IMAGES")
    print("=" * 60)
    
    images_dir = Path("datasets/datasets")
    labels_dir = Path("datasets/labels")
    
    image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
    
    print(f"\n📊 Found {len(image_files)} images")
    
    for img_path in image_files:
        # Check if image is readable
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"   ❌ Cannot read: {img_path.name}")
            continue
        
        # Check if corresponding label exists
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            print(f"   ⚠️ Missing label: {img_path.name}")
            continue
        
        # Check label content
        with open(label_path, 'r') as f:
            lines = [l for l in f.readlines() if l.strip()]
        
        h, w = img.shape[:2]
        print(f"   ✅ {img_path.name}: {w}x{h}, {len(lines)} label(s)")

def main():
    """Run all diagnostics and fixes"""
    print("\n" + "=" * 60)
    print("🏏 SPONSOR DETECTION - DIAGNOSTIC & FIX TOOL")
    print("=" * 60)
    
    # Step 1: Check training images
    check_training_images()
    
    # Step 2: Check labels
    issues = check_labels()
    
    # Step 3: Fix labels if issues found
    if issues:
        print(f"\n⚠️ Found {len(issues)} label issues!")
        response = input("\n🔧 Fix labels automatically? (y/n): ")
        if response.lower() == 'y':
            fix_labels()
            print("\n✅ Labels fixed! Please retrain the model:")
            print("   python train_sponsor_model.py")
        else:
            print("   Skipping fixes.")
    else:
        print("\n✅ All labels are correct!")
    
    # Step 4: Test detection
    print("\n" + "=" * 60)
    test_detection_with_low_confidence()
    
    print("\n" + "=" * 60)
    print("🎯 RECOMMENDATIONS")
    print("=" * 60)
    print("""
1. If labels were fixed, RETRAIN the model:
   python train_sponsor_model.py

2. If still no detections:
   - Dataset is too small (only 9 images)
   - Add 50-100 more images per sponsor
   - Use actual video frames with sponsors visible

3. Lower confidence threshold in app.py:
   - Change confidence_threshold from 0.5 to 0.1

4. Send me a sample video frame and I can help debug further
    """)

if __name__ == "__main__":
    main()
