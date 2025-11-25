"""
Workaround for Windows path bug - copy model to safe location
"""

from pathlib import Path
import shutil
from ultralytics import YOLO
import cv2

def setup_safe_model():
    """Copy model to a path without special characters"""
    
    # Source model (with apostrophe in path)
    source = Path("runs/detect/sponsor_detector_v3/weights/best.pt")
    
    # Safe destination (no special characters)
    safe_dir = Path("models")
    safe_dir.mkdir(exist_ok=True)
    dest = safe_dir / "sponsor_best.pt"
    
    if source.exists():
        shutil.copy2(source, dest)
        print(f"✅ Copied model to safe location: {dest}")
        return dest
    else:
        print(f"❌ Source model not found: {source}")
        return None

def test_model():
    """Test the model from safe location"""
    
    print("=" * 60)
    print("🧪 TESTING SPONSOR DETECTION MODEL")
    print("=" * 60)
    
    # Setup safe model path
    model_path = setup_safe_model()
    
    if not model_path:
        return
    
    # Load model
    print(f"\n📦 Loading model from: {model_path}")
    model = YOLO(str(model_path))
    
    print(f"📊 Classes: {list(model.names.values())}\n")
    
    # Test on images
    test_images = list(Path("datasets/datasets").glob("frame_*.jpg"))[:5]
    
    if not test_images:
        print("❌ No test images!")
        return
    
    print(f"🔍 Testing on {len(test_images)} frames...\n")
    
    total = 0
    
    for img_path in test_images:
        print(f"📸 {img_path.name}")
        
        results = model(str(img_path), conf=0.15, verbose=False)
        
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                print(f"   ✅ {len(boxes)} detection(s):")
                for box in boxes:
                    cls = model.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    print(f"      - {cls}: {conf:.1%}")
                    total += 1
                
                # Save result
                annotated = result.plot()
                cv2.imwrite(f"result_{img_path.stem}.jpg", annotated)
            else:
                print(f"   ⚠️ No detections")
        print()
    
    print("=" * 60)
    print(f"📊 TOTAL: {total} detections")
    print("=" * 60)
    
    if total > 0:
        print("\n✅ Model is working!")
    else:
        print("\n⚠️ No detections - may need more training data")

if __name__ == "__main__":
    test_model()
