# Jio Hotstar AdVision & Analytics System

A computer vision-based system for analyzing brand visibility and sponsorship analytics in cricket match videos using **custom-trained YOLOv8 models**.

## 🎯 Features

- ✅ **Custom Sponsor Detection** - Trained on 6 cricket sponsor brands
- ✅ **Video Upload & Processing** - Analyze cricket match videos
- ✅ **Brand Analytics Dashboard** - Visualize sponsor visibility metrics
- ✅ **Interactive Charts** - Plotly-based analytics
- ✅ **AI Assistant** - Gemini-powered chatbot for insights
- ✅ **Data Export** - Download detection data as CSV

## 🏏 Detected Sponsors

The custom model can detect these sponsors:
1. **Aramco** (Energy)
2. **DP World** (Logistics)
3. **Emirates** (Airlines)
4. **Google** (Technology)
5. **Rexona** (Personal Care)
6. **Royal Stag** (Beverage)

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
streamlit run app.py
```

### 3. Upload & Analyze
- Navigate to "Upload & Process"
- Upload a cricket match video
- View analytics and sponsor visibility metrics

## 🔧 Custom Model Training

The system includes a **custom-trained YOLOv8 model** for sponsor detection.

### Training Your Own Model

1. **Add Training Images**
   - Place sponsor logo images in `datasets/datasets/`
   - Run annotation tool: `python datasets/train.py`

2. **Train the Model**
   ```bash
   python train_sponsor_model.py
   ```

3. **Model Auto-Integration**
   - Trained model saved to `runs/detect/sponsor_detector/weights/best.pt`
   - App automatically uses the custom model

### Training Documentation
- 📖 **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Detailed training instructions
- 🎉 **[TRAINING_COMPLETE.md](TRAINING_COMPLETE.md)** - Training status & results

## 📊 Project Structure

```
jiohotstar/
├── app.py                      # Main Streamlit application
├── brand_detector.py           # YOLOv8 detection engine (custom model support)
├── video_processor.py          # Frame extraction
├── analytics.py                # Visualization & metrics
├── data_manager.py             # SQLite & CSV storage
├── train_sponsor_model.py      # Model training pipeline
├── test_model.py               # Model testing script
├── datasets/
│   ├── datasets/               # Training images
│   ├── labels/                 # YOLO annotations
│   ├── dataset.yaml            # YOLO config
│   └── train.py                # Annotation tool
└── runs/detect/sponsor_detector/
    └── weights/best.pt         # Trained model
```

## 🎨 Technology Stack

- **Frontend**: Streamlit
- **Computer Vision**: YOLOv8 (Ultralytics), OpenCV
- **Data**: Pandas, SQLite
- **Visualization**: Plotly, Matplotlib
- **AI**: Google Gemini API (chatbot)

## 📈 How It Works

1. **Upload Video** → Cricket match footage
2. **Frame Extraction** → Extract frames at intervals
3. **Sponsor Detection** → YOLOv8 detects trained logos
4. **Analytics** → Generate visibility metrics & charts
5. **Export Data** → Download results as CSV

## 🔑 Configuration

### Gemini API (Optional)
For the AI Assistant feature, add your API key to `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your_api_key_here"
```

## 📝 Notes

- Current model trained on 9 images (limited accuracy)
- For production use, collect 100+ images per sponsor
- GPU recommended for faster training
- Model works best on similar cricket footage

## 🎯 Future Improvements

- [ ] Expand training dataset (100+ images per sponsor)
- [ ] Add more sponsor brands
- [ ] Implement video clip extraction
- [ ] Real-time video processing
- [ ] Advanced analytics (ROI metrics, heatmaps)

---

**Made with ❤️ for cricket sponsor analytics**
