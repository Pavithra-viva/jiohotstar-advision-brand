"""
YOLOv8 Training Script for Cricket Sponsor Detection
This script prepares the dataset and trains a custom YOLOv8 model
"""

import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import yaml

def prepare_dataset():
    """Organize dataset into YOLO format structure"""
    print("📁 Preparing dataset structure...")
    
    # Create directory structure
    base_dir = Path("datasets")
    images_dir = base_dir / "images"
    labels_dir_new = base_dir / "labels_yolo"
    
    # Create train/val split directories
    for split in ['train', 'val']:
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir_new / split).mkdir(parents=True, exist_ok=True)
    
    # Get all images
    source_images = base_dir / "datasets"
    source_labels = base_dir / "labels"
    
    image_files = list(source_images.glob("*.png")) + list(source_images.glob("*.jpg"))
    
    print(f"📊 Found {len(image_files)} images")
    
    # Split: 80% train, 20% val
    split_idx = int(len(image_files) * 0.8)
    train_images = image_files[:split_idx]
    val_images = image_files[split_idx:]
    
    print(f"✂️ Split: {len(train_images)} train, {len(val_images)} val")
    
    # Copy files to appropriate directories
    for img_list, split in [(train_images, 'train'), (val_images, 'val')]:
        for img_path in img_list:
            # Copy image
            dest_img = images_dir / split / img_path.name
            shutil.copy2(img_path, dest_img)
            
            # Copy corresponding label
            label_name = img_path.stem + ".txt"
            src_label = source_labels / label_name
            
            if src_label.exists():
                dest_label = labels_dir_new / split / label_name
                shutil.copy2(src_label, dest_label)
                print(f"✅ Copied {img_path.name} → {split}")
    
    print("✅ Dataset preparation complete!\n")
    return images_dir, labels_dir_new

def update_dataset_yaml(images_dir, labels_dir):
    """Update dataset.yaml with correct paths"""
    yaml_path = Path("datasets/dataset.yaml")
    
    config = {
        'path': str(Path('datasets').absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: 'aramco',
            1: 'dpworld', 
            2: 'emirates',
            3: 'google',
            4: 'rexona',
            5: 'royalstag'
        }
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"📝 Updated {yaml_path}")

def train_model(epochs=50, img_size=640, batch_size=8):
    """Train YOLOv8 model on sponsor dataset"""
    print("\n🚀 Starting YOLOv8 training...\n")
    
    # Initialize model (using YOLOv8 nano for faster training)
    model = YOLO('yolov8n.pt')  # Start with pretrained weights
    
    # Use Path object to handle Windows paths with special characters
    dataset_yaml = Path('datasets/dataset.yaml').absolute()
    
    # Train the model - disable val to avoid path issues
    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        name='sponsor_detector',
        patience=10,
        save=True,
        plots=True,
        verbose=True,
        device='cpu',
        val=False  # Disable validation to avoid Windows path bug with apostrophe
    )
    
    print("\n✅ Training complete!")
    print(f"📦 Model saved to: runs/detect/sponsor_detector/weights/best.pt")
    
    return results

def validate_model():
    """Validate the trained model"""
    print("\n🔍 Validating model...")
    
    model_path = "runs/detect/sponsor_detector/weights/best.pt"
    if not os.path.exists(model_path):
        print("❌ Trained model not found!")
        return
    
    model = YOLO(model_path)
    metrics = model.val()
    
    print(f"\n📊 Validation Results:")
    print(f"   mAP50: {metrics.box.map50:.3f}")
    print(f"   mAP50-95: {metrics.box.map:.3f}")
    
    return metrics

def main():
    """Main training pipeline"""
    print("=" * 60)
    print("🏏 YOLOv8 Cricket Sponsor Detection - Training Pipeline")
    print("=" * 60)
    
    # Step 1: Prepare dataset
    images_dir, labels_dir = prepare_dataset()
    
    # Step 2: Update YAML config
    update_dataset_yaml(images_dir, labels_dir)
    
    # Step 3: Train model
    print("\n" + "=" * 60)
    print("Starting training (this may take a while)...")
    print("=" * 60 + "\n")
    
    train_model(epochs=100, img_size=640, batch_size=8)
    
    # Step 4: Validate
    validate_model()
    
    print("\n" + "=" * 60)
    print("🎉 Training pipeline complete!")
    print("=" * 60)
    print("\n📌 Next steps:")
    print("   1. Check training results in: runs/detect/sponsor_detector/")
    print("   2. Best model saved at: runs/detect/sponsor_detector/weights/best.pt")
    print("   3. Update brand_detector.py to use the new model")
    print("\n")

if __name__ == "__main__":
    main()
