import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class AnalyticsEngine:
    """Generate analytics and visualizations"""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
    
    def generate_brand_visibility_chart(self):
        """Generate brand visibility duration chart"""
        df = self.data_manager.load_detections()
        if df.empty:
            return None
            
        # Calculate visibility duration for each brand
        brand_duration = df.groupby('brand_name')['timestamp'].agg(['min', 'max'])
        brand_duration['duration'] = brand_duration['max'] - brand_duration['min']
        brand_duration = brand_duration.sort_values('duration', ascending=True)
        
        fig = px.bar(
            x=brand_duration['duration'],
            y=brand_duration.index,
            orientation='h',
            title='Brand Visibility Duration',
            labels={'x': 'Duration (seconds)', 'y': 'Brand'}
        )
        
        return fig
    
    def generate_detection_frequency_chart(self):
        """Generate detection frequency chart"""
        df = self.data_manager.load_detections()
        if df.empty:
            return None
            
        brand_counts = df['brand_name'].value_counts()
        
        fig = px.pie(
            values=brand_counts.values,
            names=brand_counts.index,
            title='Brand Detection Frequency'
        )
        
        return fig
    
    def generate_placement_analysis(self):
        """Generate placement zone analysis"""
        df = self.data_manager.load_detections()
        if df.empty:
            return None
            
        placement_brand = df.groupby(['placement', 'brand_name']).size().reset_index(name='count')
        
        fig = px.bar(
            placement_brand,
            x='placement',
            y='count',
            color='brand_name',
            title='Brand Placement Analysis',
            labels={'count': 'Number of Detections', 'placement': 'Placement Zone'}
        )
        
        return fig
    
    def generate_timeline_chart(self):
        """Generate timeline visualization"""
        df = self.data_manager.load_detections()
        if df.empty:
            return None
            
        # Create timeline plot
        fig = px.scatter(
            df,
            x='timestamp',
            y='brand_name',
            size='confidence',
            color='brand_name',
            title='Brand Appearance Timeline',
            labels={'timestamp': 'Time (seconds)', 'brand_name': 'Brand'}
        )
        
        return fig
    
    def generate_summary_stats(self) -> Dict:
        """Generate summary statistics"""
        df = self.data_manager.load_detections()
        if df.empty:
            return {}
            
        stats = {
            'total_detections': len(df),
            'unique_brands': df['brand_name'].nunique(),
            'avg_confidence': df['confidence'].mean(),
            'total_video_duration': df['timestamp'].max(),
            'most_visible_brand': df['brand_name'].value_counts().index[0] if len(df) > 0 else 'None',
            'placement_distribution': df['placement'].value_counts().to_dict()
        }
        
        return stats

    def get_brand_visibility_summary(self) -> pd.DataFrame:
        df = self.data_manager.load_detections()
        if df.empty:
            return pd.DataFrame()

        grouped = df.groupby('brand_name')['timestamp']
        summary_df = grouped.agg(['min', 'max', 'count']).reset_index()
        summary_df.rename(
            columns={
                'min': 'first_seen_sec',
                'max': 'last_seen_sec',
                'count': 'total_detections'
            },
            inplace=True,
        )

        def _format_mm_ss(seconds: float) -> str:
            seconds = max(float(seconds), 0.0)
            minutes, secs = divmod(seconds, 60)
            return f"{int(minutes):02d}:{int(secs):02d}"

        summary_df['first_seen_mmss'] = summary_df['first_seen_sec'].apply(_format_mm_ss)
        summary_df['last_seen_mmss'] = summary_df['last_seen_sec'].apply(_format_mm_ss)

        summary_df['visible_duration_sec'] = (
            summary_df['last_seen_sec'] - summary_df['first_seen_sec']
        )
        summary_df['visible_duration_mmss'] = summary_df['visible_duration_sec'].apply(
            _format_mm_ss
        )

        summary_df = summary_df[[
            'brand_name',
            'first_seen_sec',
            'first_seen_mmss',
            'last_seen_sec',
            'last_seen_mmss',
            'total_detections',
            'visible_duration_sec',
            'visible_duration_mmss',
        ]]

        return summary_df
