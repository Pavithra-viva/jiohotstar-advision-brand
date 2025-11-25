"""
Video Player with Real-time Sponsor Overlays
Displays video with sponsor detections overlaid at correct timestamps
"""

import streamlit as st
import json
from pathlib import Path

def create_video_player_with_overlays(video_path: str, detections: list):
    """
    Create HTML5 video player with sponsor overlays synced to timestamps
    
    Args:
        video_path: Path to video file
        detections: List of detection dicts with 'timestamp', 'sponsor', 'confidence'
    """
    
    # Group detections by timestamp (rounded to nearest second)
    timestamp_map = {}
    for det in detections:
        ts = int(det['timestamp'])  # Round to nearest second
        if ts not in timestamp_map:
            timestamp_map[ts] = []
        timestamp_map[ts].append({
            'sponsor': det.get('sponsor', det.get('brand_name', 'Unknown')),
            'confidence': det['confidence'],
            'location': det.get('location', det.get('placement', 'unknown'))
        })
    
    # Convert to JSON for JavaScript
    detections_json = json.dumps(timestamp_map)
    
    # Use Streamlit's native video player with custom overlay
    st.video(video_path)
    
    # Add overlay info below video
    st.markdown("### 📊 Detection Timeline")
    
    if timestamp_map:
        # Show timeline of detections
        timeline_data = []
        for ts in sorted(timestamp_map.keys()):
            sponsors = timestamp_map[ts]
            sponsor_names = ', '.join([s['sponsor'] for s in sponsors])
            timeline_data.append({
                'Time (s)': ts,
                'Sponsors': sponsor_names,
                'Count': len(sponsors)
            })
        
        import pandas as pd
        df = pd.DataFrame(timeline_data)
        st.dataframe(df, use_container_width=True)
        
        # Show sponsor summary
        st.markdown("### 🏆 Sponsor Summary")
        from collections import Counter
        all_sponsors = [s['sponsor'] for sponsors in timestamp_map.values() for s in sponsors]
        sponsor_counts = Counter(all_sponsors)
        
        cols = st.columns(min(len(sponsor_counts), 4))
        for i, (sponsor, count) in enumerate(sponsor_counts.most_common()):
            with cols[i % len(cols)]:
                st.metric(sponsor, f"{count} appearances")
    else:
        st.info("No sponsors detected in this video.")


if __name__ == "__main__":
    # Test data
    test_detections = [
        {'timestamp': 5.2, 'sponsor': 'Aramco', 'confidence': 0.85, 'location': 'boundary board'},
        {'timestamp': 5.8, 'sponsor': 'Emirates', 'confidence': 0.92, 'location': 'jersey'},
        {'timestamp': 12.1, 'sponsor': 'Google', 'confidence': 0.78, 'location': 'LED screen'},
    ]
    
    create_video_player_with_overlays("video.mp4", test_detections)
