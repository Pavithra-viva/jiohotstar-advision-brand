"""
Quick test script to verify the trained sponsor detection model
"""

from ultralytics import YOLO
import cv2
import os

def test_model():
    """Test the trained model on the training images"""
    
    model_path = "runs/detect/sponsor_detector/weights/best.pt"
    
    if not os.path.exists(model_path):
        print("❌ Model not found! Please train the model first.")
        return
    
    print("✅ Loading trained sponsor detection model...")
    model = YOLO(model_path)
    
    # Test on a sample image
    test_image = "datasets/datasets/aramco.png"
    
    if os.path.exists(test_image):
        print(f"\n🔍 Testing on: {test_image}")
        results = model(test_image, conf=0.25)
        
        # Display results
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                print(f"\n✅ Detected {len(boxes)} sponsor(s):")
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = model.names[class_id]
                    print(f"   - {class_name}: {confidence:.2%} confidence")
            else:
                print("⚠️ No sponsors detected")
        
        # Save annotated image
        annotated = results[0].plot()
        output_path = "test_detection_result.jpg"
        cv2.imwrite(output_path, annotated)
        print(f"\n💾 Saved annotated image to: {output_path}")
    else:
        print(f"❌ Test image not found: {test_image}")
    
    # Print model info
    print(f"\n📊 Model Information:")
    print(f"   Classes: {list(model.names.values())}")
    print(f"   Total classes: {len(model.names)}")

if __name__ == "__main__":
    test_model()
