import pandas as pd
import sqlite3
import os
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class DataManager:
    """Handles data storage and retrieval"""
    
    def __init__(self, db_path: str = "detections.db"):
        self.db_path = db_path
        self.csv_path = "detections.csv"
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                frame_number INTEGER,
                timestamp REAL,
                brand_name TEXT,
                confidence REAL,
                bbox_x1 REAL,
                bbox_y1 REAL,
                bbox_x2 REAL,
                bbox_y2 REAL,
                area REAL,
                placement TEXT,
                category TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_detections(self, detections: List[Dict]):
        """Save detections to both CSV and SQLite"""
        if not detections:
            return
            
        # Convert to DataFrame
        df_data = []
        for det in detections:
            row = {
                'frame_number': det['frame_number'],
                'timestamp': det['timestamp'],
                'brand_name': det['brand_name'],
                'confidence': det['confidence'],
                'bbox_x1': det['bbox'][0],
                'bbox_y1': det['bbox'][1],
                'bbox_x2': det['bbox'][2],
                'bbox_y2': det['bbox'][3],
                'area': det['area'],
                'placement': det['placement'],
                'category': det['category']
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Save to CSV
        df.to_csv(self.csv_path, index=False)
        
        # Save to SQLite
        conn = sqlite3.connect(self.db_path)
        df.to_sql('detections', conn, if_exists='replace', index=False)
        conn.close()
        
        logger.info(f"Saved {len(detections)} detections to database")
    
    def load_detections(self) -> pd.DataFrame:
        """Load detections from CSV"""
        if os.path.exists(self.csv_path):
            return pd.read_csv(self.csv_path)
        return pd.DataFrame()
    
    def get_brand_summary(self) -> Dict:
        """Get summary statistics by brand"""
        df = self.load_detections()
        if df.empty:
            return {}
            
        summary = {}
        for brand in df['brand_name'].unique():
            brand_data = df[df['brand_name'] == brand]
            summary[brand] = {
                'total_detections': len(brand_data),
                'avg_confidence': brand_data['confidence'].mean(),
                'total_duration': brand_data['timestamp'].max() - brand_data['timestamp'].min(),
                'placements': brand_data['placement'].value_counts().to_dict()
            }
        
        return summary
