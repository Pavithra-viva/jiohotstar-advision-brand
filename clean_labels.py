"""
Clean up bad labels (tiny boxes, invalid coordinates)
"""

from pathlib import Path
import shutil

def clean_labels():
    """Remove tiny/invalid bounding boxes from label files"""
    
    labels_dir = Path("datasets/labels")
    backup_dir = Path("datasets/labels_backup")
    
    # Create backup
    if not backup_dir.exists():
        shutil.copytree(labels_dir, backup_dir)
        print(f"✅ Created backup: {backup_dir}/")
    
    print("=" * 60)
    print("🧹 CLEANING LABEL FILES")
    print("=" * 60)
    
    total_boxes_before = 0
    total_boxes_after = 0
    files_cleaned = 0
    
    # Process each label file
    for label_file in labels_dir.glob("*.txt"):
        if label_file.name.endswith('.cache'):
            continue
        
        with open(label_file, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        
        total_boxes_before += len(lines)
        
        # Filter out bad boxes
        good_lines = []
        removed_count = 0
        
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                removed_count += 1
                continue
            
            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Remove tiny boxes (likely labeling errors)
                if width < 0.01 or height < 0.01:
                    print(f"   ⚠️ Removed tiny box from {label_file.name}: {width:.4f}x{height:.4f}")
                    removed_count += 1
                    continue
                
                # Remove boxes with invalid coordinates
                if width > 1.0 or height > 1.0 or x_center > 1.0 or y_center > 1.0:
                    print(f"   ⚠️ Removed invalid box from {label_file.name}")
                    removed_count += 1
                    continue
                
                # Remove boxes outside image bounds
                if x_center < 0 or y_center < 0:
                    print(f"   ⚠️ Removed out-of-bounds box from {label_file.name}")
                    removed_count += 1
                    continue
                
                good_lines.append(line)
                
            except ValueError:
                print(f"   ⚠️ Removed malformed line from {label_file.name}")
                removed_count += 1
                continue
        
        total_boxes_after += len(good_lines)
        
        # Write cleaned labels back
        if removed_count > 0:
            with open(label_file, 'w') as f:
                for line in good_lines:
                    f.write(line + '\n')
            files_cleaned += 1
            print(f"✅ Cleaned {label_file.name}: removed {removed_count} bad box(es)")
    
    print("\n" + "=" * 60)
    print("📊 CLEANING SUMMARY")
    print("=" * 60)
    print(f"Files processed: {len(list(labels_dir.glob('*.txt')))}")
    print(f"Files cleaned: {files_cleaned}")
    print(f"Boxes before: {total_boxes_before}")
    print(f"Boxes after: {total_boxes_after}")
    print(f"Boxes removed: {total_boxes_before - total_boxes_after}")
    print(f"\n✅ Labels cleaned! Backup saved to: {backup_dir}/")
    print("\n📌 Next step: Retrain the model")
    print("   python train_sponsor_model.py")

if __name__ == "__main__":
    clean_labels()
