"""
Debug Model - Test on Training Data with Ultra-Low Confidence
"""

from ultralytics import YOLO
from pathlib import Path
import cv2
import shutil

def debug_model():
    print("=" * 60)
    print("🐞 DEBUGGING MODEL")
    print("=" * 60)
    
    # 1. Load Model (Safe Copy)
    model_path = Path("runs/detect/sponsor_detector/weights/best.pt")
    safe_path = "debug_model.pt"
    
    if not model_path.exists():
        print("❌ Model not found!")
        return
        
    shutil.copy2(model_path, safe_path)
    model = YOLO(safe_path)
    
    # 2. Get a Training Image
    train_images = list(Path("datasets/labels_yolo/train").glob("*.txt"))
    if not train_images:
        print("❌ No training labels found!")
        return
        
    # Get the corresponding image for the first label
    label_path = train_images[0]
    img_name = label_path.stem + ".jpg" # Assuming jpg
    img_path = Path("datasets/images/train") / img_name
    
    if not img_path.exists():
        print(f"❌ Image not found: {img_path}")
        return
        
    print(f"📸 Testing on training image: {img_path.name}")
    print(f"📄 Label file: {label_path.name}")
    
    # Read label content to see what SHOULD be there
    with open(label_path, 'r') as f:
        print("   Expected Labels:")
        for line in f:
            print(f"   - {line.strip()}")
            
    # 3. Run Inference with varying confidence
    for conf in [0.25, 0.1, 0.01, 0.001]:
        print(f"\n🔍 Testing with conf={conf}...")
        results = model(str(img_path), conf=conf, verbose=False)
        
        detections = len(results[0].boxes)
        print(f"   Found {detections} detections")
        
        if detections > 0:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                score = float(box.conf[0])
                print(f"   - Class {cls} ({model.names[cls]}): {score:.4f}")
            
            # Save debug image
            res_plot = results[0].plot()
            cv2.imwrite(f"debug_result_conf_{conf}.jpg", res_plot)
            print(f"   💾 Saved debug_result_conf_{conf}.jpg")

if __name__ == "__main__":
    debug_model()
