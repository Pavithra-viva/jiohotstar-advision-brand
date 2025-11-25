import streamlit as st
import os
import tempfile
from pathlib import Path
from video_processor import VideoProcessor
from brand_detector import BrandDetector
from data_manager import DataManager
from clip_extractor import ClipExtractor
from analytics import AnalyticsEngine
import pandas as pd
import logging
import time
import glob
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Jio Hotstar AdVision & Analytics",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'detections' not in st.session_state:
    st.session_state.detections = []
if 'chatbot_open' not in st.session_state:
    st.session_state.chatbot_open = False
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []

# Configure Gemini API
try:
    genai.configure(api_key=st.secrets["gemini"]["api_key"])
except:
    # Fallback to hardcoded key
    genai.configure(api_key="AIzaSyBJkNDE85DZRj6xy9rxi61QZcefNnkDFBA")





def get_project_context():
    """Reads the content of all python files in the project to build a context for the chatbot."""
    context = ""
    for file_path in glob.glob("*.py"):
        with open(file_path, "r", encoding="utf-8") as f:
            context += f"--- {file_path} ---\n{f.read()}\n\n"
    return context

def render_floating_chatbot_icon():
    """Render floating chatbot icon with pure HTML/CSS."""
    st.markdown("""
    <style>
    .chatbot-icon {
        position: fixed;
        bottom: 20px;
        right: 20px;
        width: 60px;
        height: 60px;
        background-color: #25D366;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        z-index: 1000;
        transition: transform 0.2s;
    }
    .chatbot-icon:hover {
        transform: scale(1.1);
    }
    .chatbot-icon svg {
        width: 30px;
        height: 30px;
        fill: white;
    }
    </style>
    
    <a href="?page=AdVision%20Assistant" target="_self">
        <div class="chatbot-icon">
            <svg viewBox="0 0 24 24">
                <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2z"/>
            </svg>
        </div>
    </a>
    """, unsafe_allow_html=True)

def main():
    st.title("🏏 Jio Hotstar AdVision & Analytics")
    st.markdown("**AI-Powered Brand Visibility Analytics for Cricket Matches**")
    
    # Handle page navigation from URL
    query_params = st.experimental_get_query_params()
    default_page = query_params.get("page", ["Upload & Process"])[0]
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Upload & Process", "Analytics Dashboard", "Brand Clips", "Detection Data", "AdVision Assistant", "Model Settings"],
        index=["Upload & Process", "Analytics Dashboard", "Brand Clips", "Detection Data", "AdVision Assistant", "Model Settings"].index(default_page) if default_page in ["Upload & Process", "Analytics Dashboard", "Brand Clips", "Detection Data", "AdVision Assistant", "Model Settings"] else 0
    )
    
    if page == "Upload & Process":
        upload_and_process_page()
    elif page == "Analytics Dashboard":
        analytics_dashboard_page()
    elif page == "Brand Clips":
        brand_clips_page()
    elif page == "Detection Data":
        detection_data_page()
    elif page == "AdVision Assistant":
        chatbot_page()
    elif page == "Model Settings":
        model_settings_page()
    
    render_floating_chatbot_icon()

def model_settings_page():
    st.header("⚙️ Model Settings")
    
    st.subheader("Available Models")
    
    from pathlib import Path
    import datetime
    
    models_dir = Path("runs/detect")
    if not models_dir.exists():
        st.warning("No trained models found.")
        return

    # Find all best.pt files
    model_files = list(models_dir.glob("*/weights/best.pt"))
    
    if not model_files:
        st.info("No trained models found yet.")
        return
        
    # Create a table of models
    model_data = []
    for model_path in model_files:
        stats = model_path.stat()
        created_time = datetime.datetime.fromtimestamp(stats.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
        size_mb = stats.st_size / (1024 * 1024)
        
        model_data.append({
            "Model Name": model_path.parent.parent.name,
            "Created": created_time,
            "Size": f"{size_mb:.2f} MB",
            "Path": str(model_path)
        })
    
    st.dataframe(model_data, use_container_width=True)
    
    # Selection
    model_names = [m["Model Name"] for m in model_data]
    selected_model = st.selectbox("Select Active Model", model_names)
    
    if st.button("Save Selection"):
        st.session_state['active_model'] = selected_model
        st.success(f"Active model set to: {selected_model}")
        
    st.info(f"Current Active Model: {st.session_state.get('active_model', 'Auto-detect (Latest)')}")
def upload_and_process_page():
    st.header("📹 Video Upload & Processing")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Cricket Match Video",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a cricket match video for brand detection analysis"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        st.success(f"✅ Video uploaded: {uploaded_file.name}")
        
        # Processing parameters
        st.subheader("Processing Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            frame_interval = st.slider("Frame Extraction Interval", 15, 60, 30, 
                                     help="Extract every nth frame (lower = more frames)")
        
        # Confidence threshold
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.01,
            max_value=1.0,
            value=0.15,
            step=0.01,
            help="Lower this if detections are missed. Increase if seeing false positives."
        )
        
        # AI Analysis Limit
        max_ai_frames = st.slider(
            "AI Analysis Frames",
            min_value=2,
            max_value=10,
            value=6,
            step=2,
            help="⚠️ CAUTION: Free tier has strict limits! Start with 2-6 frames. Higher = more API calls = quota risk!"
        )
        
        # Process button
        if st.button("🚀 Start Processing", type="primary"):
            process_video(video_path, frame_interval, confidence_threshold, max_ai_frames)

def process_video(video_path: str, frame_interval: int, confidence_threshold: float, max_ai_frames: int = 15):
    """Process video with progress tracking using Gemini AI"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize components
        video_processor = VideoProcessor(frame_interval=frame_interval)
        data_manager = DataManager()
        
        # Step 1: Extract frames
        status_text.text("🎬 Extracting frames from video...")
        progress_bar.progress(20)
        
        frame_paths, video_info = video_processor.extract_frames(video_path)
        st.info(f"📊 Extracted {len(frame_paths)} frames from {video_info['duration']:.1f}s video")
        
        # Step 2: Detect brands using Gemini (with smart sampling)
        status_text.text("🔍 Analyzing frames with AI...")
        progress_bar.progress(40)
        
        from gemini_detector import GeminiSponsorDetector
        import random
        
        try:
            api_key = st.secrets["gemini"]["api_key"]
        except:
            api_key = "AIzaSyBJkNDE85DZRj6xy9rxi61QZcefNnkDFBA"
        
        detector = GeminiSponsorDetector(api_key)
        
        # HARD LIMIT: Never process more than 10 frames, regardless of user input
        ABSOLUTE_MAX_FRAMES = 10
        max_frames_to_analyze = min(ABSOLUTE_MAX_FRAMES, max_ai_frames, len(frame_paths))
        
        if len(frame_paths) > max_frames_to_analyze:
            # Evenly distribute samples across video duration
            step = len(frame_paths) // max_frames_to_analyze
            sampled_indices = [i * step for i in range(max_frames_to_analyze)]
            sampled_frames = [frame_paths[i] for i in sampled_indices]
        else:
            sampled_frames = frame_paths[:max_frames_to_analyze]  # Take only first N frames
        
        # Process sampled frames with Gemini (WITH RATE LIMITING!)
        all_detections = []
        import time
        
        for i, frame_path in enumerate(sampled_frames):
            timestamp = video_processor.get_frame_timestamp(frame_path)
            
            # Update progress silently
            status_text.text(f"🔍 Analyzing video content... ({i+1}/{len(sampled_frames)})")
            
            try:
                detections = detector.detect_in_frame(frame_path, timestamp)
                
                # Filter by confidence and add ALL missing fields for DataManager
                for det in detections:
                    if det['confidence'] >= confidence_threshold:
                        # Add fields expected by DataManager
                        det['frame_number'] = i
                        det['brand_name'] = det['sponsor']  # Map sponsor to brand_name
                        det['category'] = 'Sponsor'  # Generic category
                        det['placement'] = det.get('location', 'unknown')
                        
                        # Gemini doesn't provide bounding boxes, so add dummy values
                        det['bbox'] = [0, 0, 100, 100]  # Dummy bbox
                        det['center'] = [50, 50]  # Dummy center
                        det['area'] = 10000  # Dummy area
                        
                        all_detections.append(det)
                
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    status_text.text(f"⏸️ Pausing briefly...")
                    time.sleep(10)
                    # Retry once
                    try:
                        detections = detector.detect_in_frame(frame_path, timestamp)
                        for det in detections:
                            if det['confidence'] >= confidence_threshold:
                                det['frame_number'] = i
                                det['brand_name'] = det['sponsor']
                                det['category'] = 'Sponsor'
                                det['placement'] = det.get('location', 'unknown')
                                det['bbox'] = [0, 0, 100, 100]
                                det['center'] = [50, 50]
                                det['area'] = 10000
                                all_detections.append(det)
                    except:
                        pass  # Skip silently
            
            progress = 40 + int((i + 1) / len(sampled_frames) * 40)
            progress_bar.progress(progress)
            
            # CRITICAL: Add delay after every 2 frames to respect rate limits
            if (i + 1) % 2 == 0 and i < len(sampled_frames) - 1:
                time.sleep(5)  # Wait 5 seconds after every 2 requests
        
        # Step 3: Save data
        status_text.text("💾 Saving detection data...")
        progress_bar.progress(80)
        
        if all_detections:
            data_manager.save_detections(all_detections)
            st.success(f"✅ Found {len(all_detections)} sponsor detections!")
            
            # Update session state for other pages
            st.session_state.processed = True
            st.session_state.detections = all_detections
        else:
            st.warning("⚠️ No sponsors detected. Try lowering the confidence threshold.")
        
        # Step 4: Complete
        progress_bar.progress(100)
        status_text.text("✅ Processing complete!")
        
        st.balloons()
        
        # Show video player with overlays
        if all_detections:
            st.divider()
            st.subheader("🎥 Watch Video with Sponsor Detections")
            st.markdown("The video is ready! Check the **Analytics Dashboard** and **Brand Clips** pages for detailed insights.")
            
            # Show navigation hint
            st.info("💡 **Next Steps:** Use the sidebar to navigate to:\n- 📊 **Analytics Dashboard** - View charts and statistics\n- 🎬 **Brand Clips** - Extract video clips of each sponsor")
            
            from video_player_overlay import create_video_player_with_overlays
            create_video_player_with_overlays(video_path, all_detections)
        
    except Exception as e:
        st.error(f"❌ Error processing video: {str(e)}")
        logger.error(f"Processing error: {e}", exc_info=True)

def analytics_dashboard_page():
    st.header("📊 Analytics Dashboard")
    
    if not st.session_state.processed:
        st.warning("⚠️ Please upload and process a video first!")
        return
    
    # Initialize analytics
    data_manager = DataManager()
    analytics = AnalyticsEngine(data_manager)
    
    # Summary stats
    st.subheader("📈 Summary Statistics")
    stats = analytics.generate_summary_stats()
    
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Detections", stats['total_detections'])
        with col2:
            st.metric("Unique Brands", stats['unique_brands'])
        with col3:
            st.metric("Avg Confidence", f"{stats['avg_confidence']:.2f}")
        with col4:
            st.metric("Video Duration", f"{stats['total_video_duration']:.1f}s")
        
        st.subheader("⏱ Brand Visibility (Per Sponsor)")
        visibility_summary = analytics.get_brand_visibility_summary()
        if not visibility_summary.empty:
            st.dataframe(visibility_summary, use_container_width=True)
    
    # Charts
    st.subheader("📊 Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Brand visibility chart
        visibility_fig = analytics.generate_brand_visibility_chart()
        if visibility_fig:
            st.plotly_chart(visibility_fig, use_container_width=True)
    
    with col2:
        # Detection frequency chart
        frequency_fig = analytics.generate_detection_frequency_chart()
        if frequency_fig:
            st.plotly_chart(frequency_fig, use_container_width=True)
    
    # Timeline chart
    timeline_fig = analytics.generate_timeline_chart()
    if timeline_fig:
        st.plotly_chart(timeline_fig, use_container_width=True)
    
    # Placement analysis
    placement_fig = analytics.generate_placement_analysis()
    if placement_fig:
        st.plotly_chart(placement_fig, use_container_width=True)

def brand_clips_page():
    st.header("🎬 Brand Video Clips")
    
    if not st.session_state.processed:
        st.warning("⚠️ Please upload and process a video first!")
        return
    
    # Show what clips would be available
    data_manager = DataManager()
    df = data_manager.load_detections()
    
    if not df.empty:
        brands = df['brand_name'].unique()
        st.subheader("Available Brand Clips:")
        
        for brand in brands:
            brand_detections = df[df['brand_name'] == brand]
            st.write(f"**{brand}**: {len(brand_detections)} clips available")

def detection_data_page():
    st.header("📄 Raw Detection Data")
    
    if not st.session_state.get('processed', False):
        st.warning("⚠️ Please upload and process a video first!")
        return

    data_manager = DataManager()
    df = data_manager.load_detections()

    if not df.empty:
        st.info(f"Displaying {len(df)} detection records.")
        st.dataframe(df, use_container_width=True)

        # Download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name='detection_data.csv',
            mime='text/csv',
        )
    else:
        st.info("No detection data available to display.")

def chatbot_page():
    st.header("🤖 AdVision Assistant")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask about the code or the latest video results..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # Check if the Gemini API key is configured
            try:
                api_key = st.secrets["gemini"]["api_key"]
                has_api_key = True
            except:
                api_key = "AIzaSyBJkNDE85DZRj6xy9rxi61QZcefNnkDFBA"  # Fallback
                has_api_key = True
            
            if not has_api_key:
                full_response = "I can't answer right now. Please make sure the Gemini API key is configured correctly in your Streamlit secrets."
                message_placeholder.markdown(full_response)
            else:
                # Build the context for the AI
                code_context = get_project_context()
                data_context = ""

                # If a video has been processed, add data summary to context
                if st.session_state.get('processed', False):
                    data_manager = DataManager()
                    analytics = AnalyticsEngine(data_manager)
                    stats = analytics.generate_summary_stats()
                    if stats:
                        data_context = f"""Here is a summary of the most recently processed video:
                        - Total Detections: {stats.get('total_detections', 'N/A')}
                        - Unique Brands: {stats.get('unique_brands', 'N/A')}
                        - Most Visible Brand: {stats.get('most_visible_brand', 'N/A')}
                        - Average Detection Confidence: {stats.get('avg_confidence', 0):.2f}
                        """
                
                # Construct the full prompt
                prompt_with_context = f"""You are AdVision Assistant, an expert on the Jio Hotstar AdVision & Analytics project.
                Your task is to answer questions. Use the DATA CONTEXT for questions about the latest video analysis results.
                Use the CODE CONTEXT for questions about how the project works.

                --- DATA CONTEXT ---
                {data_context if data_context else 'No video has been processed yet.'}

                --- CODE CONTEXT ---
                {code_context}

                --- QUESTION ---
                {prompt}
                """
                
                # Call the generative model
                model = genai.GenerativeModel('gemini-2.0-flash')
                response = model.generate_content(prompt_with_context)
                full_response = response.text
                message_placeholder.markdown(full_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
