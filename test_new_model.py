"""
Test the newly trained model (sponsor_detector2) on sample images
"""

from ultralytics import YOLO
import cv2
import os
from pathlib import Path

def test_new_model():
    """Test the latest trained model"""
    
    # Find the latest model
    model_paths = [
        "runs/detect/sponsor_detector2/weights/best.pt",
        "runs/detect/sponsor_detector/weights/best.pt"
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print("❌ No trained model found!")
        return
    
    print("=" * 60)
    print(f"🧪 TESTING MODEL: {model_path}")
    print("=" * 60)
    
    model = YOLO(model_path)
    
    print(f"\n📊 Model Classes: {list(model.names.values())}")
    
    # Test on training images
    test_images = list(Path("datasets/datasets").glob("*.png"))[:5]
    
    if not test_images:
        print("❌ No test images found!")
        return
    
    print(f"\n🔍 Testing on {len(test_images)} images...\n")
    
    total_detections = 0
    
    for img_path in test_images:
        print(f"📸 {img_path.name}")
        
        # Run detection with low confidence threshold
        results = model(str(img_path), conf=0.1, verbose=False)
        
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                print(f"   ✅ Found {len(boxes)} detection(s):")
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = model.names[class_id]
                    print(f"      - {class_name}: {confidence:.1%} confidence")
                    total_detections += 1
                
                # Save annotated image
                annotated = result.plot()
                output_path = f"test_result_{img_path.stem}.jpg"
                cv2.imwrite(output_path, annotated)
                print(f"   💾 Saved: {output_path}")
            else:
                print(f"   ⚠️ No detections")
        print()
    
    print("=" * 60)
    print(f"📊 SUMMARY: {total_detections} total detections across {len(test_images)} images")
    print("=" * 60)
    
    if total_detections == 0:
        print("\n⚠️ No detections found. This could mean:")
        print("   1. Model needs more training data")
        print("   2. Images are too different from training data")
        print("   3. Bounding boxes in labels were incorrect")
        print("\n💡 Recommendation:")
        print("   - Check if labels were created correctly")
        print("   - Add more diverse training images (50+ per class)")
        print("   - Verify bounding boxes cover the logos properly")
    else:
        print("\n✅ Model is working! Upload a cricket video to test in the app.")

if __name__ == "__main__":
    test_new_model()
