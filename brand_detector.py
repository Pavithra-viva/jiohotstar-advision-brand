import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
import logging
import os
from pathlib import Path  # Added missing import
import shutil  # Added for safe copy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BrandDetector:
    """Handles brand logo detection using YOLOv8"""
    
    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.5, allowed_classes: Optional[List[str]] = None, use_custom_model: bool = False):
        """
        Initialize BrandDetector
        
        Args:
            model_path: Path to YOLOv8 model (will download if not exists)
            confidence_threshold: Minimum confidence for detections
            allowed_classes: List of allowed class names (for base model)
            use_custom_model: If True, use custom trained sponsor detection model
        """
        self.confidence_threshold = confidence_threshold
        self.use_custom_model = use_custom_model
        
        # Use custom trained model if available
        if use_custom_model:
            # Try to find the latest trained model using Path objects
            possible_paths = [
                Path("runs/detect/sponsor_detector_v3/weights/best.pt"),
                Path("runs/detect/sponsor_detector2/weights/best.pt"),
                Path("runs/detect/sponsor_detector/weights/best.pt"),
            ]
            
            custom_model_path = None
            for path in possible_paths:
                if path.exists():
                    custom_model_path = str(path.absolute())
                    break
            
            if custom_model_path:
                # WORKAROUND: Copy model to a safe path (no apostrophes)
                safe_model_path = "temp_model_safe_detector.pt"
                try:
                    shutil.copy2(custom_model_path, safe_model_path)
                    self.model = YOLO(safe_model_path)
                    logger.info(f"✅ Loaded custom sponsor detection model from {custom_model_path} (via safe copy)")
                except Exception as e:
                    logger.error(f"❌ Error copying model: {e}")
                    self.model = YOLO(custom_model_path) # Fallback
            else:
                logger.warning(f"⚠️ Custom model not found, using base model")
                self.model = YOLO(model_path)
        else:
            self.model = YOLO(model_path)
        
        self.allowed_classes = set(allowed_classes) if allowed_classes else None
        
        # Brand mapping for base YOLO model (legacy support)
        self.brand_mapping = {}
        
        # Brand categories for analytics - Updated with actual sponsors
        self.brand_categories = {
            # Cricket Sponsors from dataset
            'aramco': 'Energy',
            'dpworld': 'Logistics',
            'emirates': 'Airlines',
            'google': 'Technology',
            'rexona': 'Personal Care',
            'royalstag': 'Beverage',
            # Legacy categories
            'Pepsi': 'Beverage',
            'MRF': 'Automotive', 
            'Jio': 'Telecom',
            'Dream11': 'Gaming',
            'Jersey_Logo': 'Apparel',
            'Ball_Sponsor': 'Equipment',
            'LED_Board': 'Stadium',
            'Digital_Ad': 'Digital',
            'Sponsor_Cup': 'Merchandise'
        }
        
        logger.info(f"BrandDetector initialized with confidence threshold: {confidence_threshold}")
        logger.info(f"Custom model enabled: {use_custom_model}")
    
    def detect_brands_in_frame(self, frame_path: str, frame_number: int, timestamp: float) -> List[Dict]:
        """
        Detect brands in a single frame
        
        Args:
            frame_path: Path to frame image
            frame_number: Frame number in sequence
            timestamp: Timestamp in video
            
        Returns:
            List of detection dictionaries
        """
        if not os.path.exists(frame_path):
            logger.warning(f"Frame not found: {frame_path}")
            return []
            
        # Load image
        image = cv2.imread(frame_path)
        if image is None:
            logger.warning(f"Cannot load image: {frame_path}")
            return []
            
        # Run YOLO detection
        results = self.model(image, conf=self.confidence_threshold, verbose=False)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get detection info
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]

                    if self.allowed_classes is not None and class_name not in self.allowed_classes:
                        continue
                    
                    # Map to brand if applicable
                    brand_name = self.brand_mapping.get(class_name, class_name)
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # Calculate center and area
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    area = (x2 - x1) * (y2 - y1)
                    
                    # Determine placement zone
                    image_height, image_width = image.shape[:2]
                    placement = self._get_placement_zone(center_x, center_y, image_width, image_height)
                    
                    detection = {
                        'frame_number': frame_number,
                        'timestamp': timestamp,
                        'frame_path': frame_path,
                        'brand_name': brand_name,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2],
                        'center': [center_x, center_y],
                        'area': area,
                        'placement': placement,
                        'category': self.brand_categories.get(brand_name, 'Other')
                    }
                    
                    detections.append(detection)
        
        return detections
    
    def _get_placement_zone(self, x: float, y: float, width: int, height: int) -> str:
        """Determine placement zone based on position"""
        # Normalize coordinates
        norm_x = x / width
        norm_y = y / height
        
        # Define zones
        if norm_y < 0.3:
            return "Top_Third"
        elif norm_y > 0.7:
            return "Bottom_Third"
        else:
            if norm_x < 0.3:
                return "Left_Side"
            elif norm_x > 0.7:
                return "Right_Side"
            else:
                return "Center"
    
    def process_frames_batch(self, frame_paths: List[str], video_processor) -> List[Dict]:
        """
        Process multiple frames for brand detection
        
        Args:
            frame_paths: List of frame file paths
            video_processor: VideoProcessor instance for timestamp extraction
            
        Returns:
            List of all detections
        """
        all_detections = []
        
        logger.info(f"Processing {len(frame_paths)} frames for brand detection...")
        
        for i, frame_path in enumerate(frame_paths):
            timestamp = video_processor.get_frame_timestamp(frame_path)
            detections = self.detect_brands_in_frame(frame_path, i, timestamp)
            all_detections.extend(detections)
            
            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i + 1}/{len(frame_paths)} frames")
        
        logger.info(f"Brand detection complete: {len(all_detections)} detections found")
        return all_detections
    
    def draw_detections_on_frame(self, frame_path: str, detections: List[Dict], output_path: str = None):
        """
        Draw detection boxes on frame for visualization
        
        Args:
            frame_path: Input frame path
            detections: List of detections for this frame
            output_path: Output path for annotated frame
        """
        image = cv2.imread(frame_path)
        if image is None:
            return
            
        for detection in detections:
            x1, y1, x2, y2 = [int(coord) for coord in detection['bbox']]
            brand_name = detection['brand_name']
            confidence = detection['confidence']
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{brand_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        if output_path:
            cv2.imwrite(output_path, image)
        
        return image
