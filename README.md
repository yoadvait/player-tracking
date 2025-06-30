# Player Tracking System

A real-time player tracking and re-identification system for sports videos using computer vision and deep learning. This system can track multiple players across frames, handle occlusions, and re-identify players when they reappear after being lost.

## Features

### Core Tracking
- Multi-object tracking with advanced association algorithms
- Robust handling of occlusions and temporary disappearances
- Adaptive IoU thresholds based on detection confidence
- Motion prediction for improved tracking stability

### Re-identification
- Deep feature extraction for player appearance modeling
- Gallery-based re-identification system with long-term memory
- Jersey number detection and verification
- Team classification based on uniform colors

### Performance Optimizations
- Lightweight feature extraction mode for real-time processing
- Configurable confidence thresholds and tracking parameters
- Efficient FAISS indexing for large-scale gallery searches
- Smart track management with confidence decay

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd player-tracking
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Put model.pt in the models folder

## Quick Start

### Basic Demo
Run the advanced tracking demo on a sample video:

```bash
python main_demo.py --input data/input/your_video.mp4 --output outputs/videos/tracked_output.mp4
```

### Configuration
The system uses YAML configuration files. The main demo uses `configs/config_demo.yaml`:

```bash
python main_demo.py --config configs/config_demo.yaml --input your_video.mp4
```

## System Architecture

### Detection
- YOLO-based player detection with confidence filtering
- Support for multiple player classes and ball detection
- Boundary filtering to remove edge detections

### Tracking
- Enhanced association using IoU, appearance, and motion cues
- Confidence-based track management with decay mechanisms
- Lost track buffer for handling temporary occlusions

### Re-identification
- Lightweight feature extraction using color histograms and texture analysis
- Gallery system for storing and matching player appearances
- Jersey number OCR for additional verification

### Visualization
- Real-time tracking visualization with player IDs
- Team color coding and confidence display
- Re-identification event highlighting

## Configuration Options

### Detection Settings
```yaml
detection:
  conf_threshold: 0.4
  iou_threshold: 0.45
  max_det: 50
```

### Tracking Parameters
```yaml
tracking:
  track_high_thresh: 0.5
  track_low_thresh: 0.1
  track_buffer: 30
  appearance_thresh: 0.3
```

### Re-identification
```yaml
advanced_reid:
  enabled: true
  feature_extractor: "lightweight"
  feature_dim: 512
  similarity_threshold: 0.6
```


## File Structure

```
player-tracking/
├── src/
│   ├── detection/     
│   ├── tracking/           
│   ├── features/    
│   ├── reidentification/ 
│   └── visualization
├── configs/         
├── data/                  
├── models/                
├── outputs/         
└── main_demo.py      
```
