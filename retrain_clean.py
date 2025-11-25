"""
Quick retrain script with cleaned labels
"""

from ultralytics import YOLO
from pathlib import Path
import shutil

def retrain_model():
    """Retrain model with cleaned labels"""
    
    print("=" * 60)
    print("🏏 RETRAINING WITH CLEANED LABELS")
    print("=" * 60)
    
    # Check if dataset is ready
    images_train = Path("datasets/images/train")
    labels_train = Path("datasets/labels_yolo/train")
    
    if not images_train.exists() or not labels_train.exists():
        print("\n📁 Preparing dataset structure...")
        from train_sponsor_model import prepare_dataset, update_dataset_yaml
        images_dir, labels_dir = prepare_dataset()
        update_dataset_yaml(images_dir, labels_dir)
    
    print("\n🚀 Starting training...")
    print("   - Model: YOLOv8n")
    print("   - Epochs: 150 (increased for better learning)")
    print("   - Batch size: 4 (reduced to avoid memory issues)")
    print("   - Image size: 640")
    print()
    
    # Initialize model
    model = YOLO('yolov8n.pt')
    
    # Train
    try:
        results = model.train(
            data=str(Path('datasets/dataset.yaml').absolute()),
            epochs=150,
            imgsz=640,
            batch=4,
            name='sponsor_detector_v3',
            patience=20,
            save=True,
            plots=True,
            verbose=True,
            device='cpu',
            exist_ok=True,
            val=False  # Disable validation to avoid Windows path bug
        )
        
        print("\n✅ Training complete!")
        print(f"📦 Model saved to: runs/detect/sponsor_detector_v3/weights/best.pt")
        
        # Test the model
        print("\n🧪 Testing model...")
        test_model = YOLO('runs/detect/sponsor_detector_v3/weights/best.pt')
        
        # Test on a sample image
        test_img = list(Path("datasets/datasets").glob("frame_*.jpg"))[0]
        results = test_model(str(test_img), conf=0.1, verbose=False)
        
        detections = 0
        for result in results:
            if result.boxes is not None:
                detections = len(result.boxes)
        
        if detections > 0:
            print(f"✅ Model detected {detections} sponsor(s) in test image!")
        else:
            print("⚠️ No detections in test image (may need more training data)")
        
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        print("\nThis might be due to:")
        print("   1. Insufficient memory")
        print("   2. Corrupted labels")
        print("   3. Path issues")

if __name__ == "__main__":
    retrain_model()
