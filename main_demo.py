import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import yaml
import cv2
import time
import logging
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any

from src.detection.detector import PlayerDetector
from src.tracking.advanced_tracker import AdvancedPlayerTracker
from src.features.team_classifier import TeamClassifier
from src.visualization.advanced_visualizer import AdvancedVisualizer

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    Path("outputs/logs").mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('outputs/logs/demo_advanced.log')
        ]
    )
    return logging.getLogger(__name__)

def make_json_serializable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_serializable(item) for item in obj)
    else:
        return obj

def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def process_demo_video(config: Dict, input_path: str, output_path: str) -> Dict:
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting Advanced Demo System with Re-identification...")
    
    detector = PlayerDetector(config)
    tracker = AdvancedPlayerTracker(config)
    team_classifier = TeamClassifier(config)
    visualizer = AdvancedVisualizer(config)
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"📹 Processing: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    start_time = time.time()
    reid_events = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame)
        team_assignments = team_classifier.classify_teams(frame, tracks)
        
        reid_stats = tracker.get_reid_statistics()
        
        reid_info = {
            'reidentified_tracks': [],
            'recent_reidentifications': reid_events[-5:],
            'stats': reid_stats,
            'gallery_size': reid_stats.get('gallery_stats', {}).get('total_entries', 0)
        }
        
        annotated = visualizer.draw_advanced_tracks(
            frame, tracks, team_assignments, reid_info, {'reid_stats': reid_stats}
        )
        out.write(annotated)
        
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps_current = frame_count / elapsed
            progress = (frame_count / total_frames) * 100
            
            logger.info(f"Progress: {progress:.1f}% | FPS: {fps_current:.1f} | "
                       f"Re-IDs: {reid_stats.get('successful_reids', 0)} | "
                       f"Jersey: {reid_stats.get('jersey_detections', 0)} | "
                       f"Gallery: {reid_stats.get('gallery_stats', {}).get('total_entries', 0)}")
    
    cap.release()
    out.release()
    
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time
    final_stats = tracker.get_reid_statistics()
    
    results = {
        'frames_processed': frame_count,
        'processing_time': total_time,
        'average_fps': avg_fps,
        'reid_stats': final_stats,
        'detection_stats': detector.get_stats() if hasattr(detector, 'get_stats') else {},
        'team_classification_stats': team_classifier.get_stats() if hasattr(team_classifier, 'get_stats') else {},
        'advanced_features': {
            'deep_features_used': True,
            'gallery_based_reid': True,
            'jersey_number_detection': reid_stats.get('jersey_detections', 0) > 0,
            'part_based_features': True
        }
    }
    
    try:
        tracker.save_reid_data("outputs/reid_data")
        logger.info("💾 Re-identification data saved to outputs/reid_data/")
    except Exception as e:
        logger.warning(f"Could not save re-ID data: {e}")
    
    logger.info(f"✅ Advanced Demo completed!")
    logger.info(f"📊 Performance: {frame_count} frames in {total_time:.1f}s ({avg_fps:.1f} FPS)")
    logger.info(f"🎯 Re-identifications: {final_stats.get('successful_reids', 0)}")
    logger.info(f"🔢 Jersey detections: {final_stats.get('jersey_detections', 0)}")
    logger.info(f"📚 Gallery entries: {final_stats.get('gallery_stats', {}).get('total_entries', 0)}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Advanced Demo Re-identification System")
    parser.add_argument("--config", default="configs/config_demo.yaml")
    parser.add_argument("--input", help="Input video")
    parser.add_argument("--output", help="Output video")
    parser.add_argument("--log-level", default="INFO")
    
    args = parser.parse_args()
    
    logger = setup_logging(args.log_level)
    config = load_config(args.config)
    
    input_path = args.input or config['video']['input_path']
    output_path = args.output or config['video']['output_path']
    
    if not Path(input_path).exists():
        logger.error(f"❌ Video not found: {input_path}")
        return
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        results = process_demo_video(config, input_path, output_path)
        

        serializable_results = make_json_serializable(results)
        
        stats_path = "outputs/demo_advanced_results.json"
        with open(stats_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n🏆 ADVANCED DEMO RESULTS:")
        print(f"   ⚡ Processing Speed: {results['average_fps']:.1f} FPS")
        print(f"   🎯 Re-identifications: {results['reid_stats']['successful_reids']}")
        print(f"   🔢 Jersey Numbers: {results['reid_stats'].get('jersey_detections', 0)}")
        print(f"   📚 Gallery Entries: {results['reid_stats'].get('gallery_stats', {}).get('total_entries', 0)}")
        print(f"   📁 Output Video: {output_path}")
        print(f"   📊 Statistics: {stats_path}")
        print(f"   💾 Re-ID Data: outputs/reid_data/")
        
    except Exception as e:
        logger.error(f"❌ Advanced Demo failed: {e}")
        raise

if __name__ == "__main__":
    main()