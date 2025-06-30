import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional

class PlayerDetector:
    
    def __init__(self, config: Dict):
        self.config = config['detection']
        self.logger = logging.getLogger(__name__)
        
        self.model = self._load_model()
        self.device = self._setup_device()
        
        self.conf_threshold = max(self.config.get('conf_threshold', 0.4), 0.4)
        self.iou_threshold = self.config['iou_threshold']
        self.max_det = self.config['max_det']
        self.classes = self.config.get('classes', None)
        
        self.class_names = {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}
        self.player_classes = [1, 2]
        self.ball_classes = [0]
        self.referee_classes = [3]
        
        self.previous_detections = {}
        self.detection_history = {}
        self.frame_count = 0
        self.total_detections = 0
        
        self.size_change_threshold = 0.3
        self.persistence_boost_factor = 0.1
        self.max_confidence_boost = 0.3
        
        self.logger.info(f"Player classes: {self.player_classes}")
        self.logger.info(f"Ball classes: {self.ball_classes}")
        self.logger.info(f"Confidence threshold set to: {self.conf_threshold}")
        
    def _load_model(self) -> YOLO:
        model_path = Path(self.config['model_path'])
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
            
        try:
            model = YOLO(str(model_path))
            self.logger.info(f"Model loaded: {model_path}")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def _setup_device(self) -> str:
        if self.config['device'] == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = self.config['device']
            
        self.logger.info(f"Using device: {device}")
        return device
    
    def detect(self, frame: np.ndarray) -> Dict:
    
        try:
            results = self.model(
                frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                max_det=self.max_det,
                classes=self.classes,
                device=self.device,
                verbose=False
            )
            
            detection_data = self._process_results(results[0], frame)
            
            self.frame_count += 1
            self.total_detections += len(detection_data['boxes'])
            
            self._update_detection_history(detection_data)
            
            return detection_data
            
        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
            return self._empty_detection()
    
    def _process_results(self, result, frame: np.ndarray) -> Dict:
        
        if result.boxes is None or len(result.boxes) == 0:
            return self._empty_detection()
        
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        
        player_mask = np.isin(classes, self.player_classes)
        
        if np.any(player_mask):
            player_boxes = boxes[player_mask]
            player_scores = scores[player_mask]
            player_classes = classes[player_mask]
            
            quality_mask = self._filter_detections_enhanced(player_boxes, player_scores, frame.shape)
            
            filtered_boxes = player_boxes[quality_mask]
            filtered_scores = player_scores[quality_mask]
            filtered_classes = player_classes[quality_mask]
            
            boosted_scores = self._apply_confidence_boosting(filtered_boxes, filtered_scores)
            
            final_boxes = filtered_boxes
            final_scores = boosted_scores
            final_classes = filtered_classes
        else:
            final_boxes = np.array([]).reshape(0, 4)
            final_scores = np.array([])
            final_classes = np.array([])
        
        features = self._extract_features(final_boxes, frame) if len(final_boxes) > 0 else None
        
        ball_mask = np.isin(classes, self.ball_classes)
        referee_mask = np.isin(classes, self.referee_classes)
        
        return {
            'boxes': final_boxes,
            'scores': final_scores,
            'features': features,
            'classes': final_classes,
            'ball_boxes': boxes[ball_mask] if np.any(ball_mask) else np.array([]).reshape(0, 4),
            'ball_scores': scores[ball_mask] if np.any(ball_mask) else np.array([]),
            'referee_boxes': boxes[referee_mask] if np.any(referee_mask) else np.array([]).reshape(0, 4),
            'referee_scores': scores[referee_mask] if np.any(referee_mask) else np.array([]),
            'frame_shape': frame.shape,
            'timestamp': self.frame_count,
            'total_detections': len(boxes),
            'player_detections': len(final_boxes)
        }
    
    def _filter_detections_enhanced(self, boxes: np.ndarray, scores: np.ndarray, 
                          frame_shape: Tuple) -> np.ndarray:
        if len(boxes) == 0:
            return np.array([], dtype=bool)
        
        h, w = frame_shape[:2]
        
        boundary_margin = 15
        boundary_filter = np.logical_and.reduce([
            boxes[:, 0] >= boundary_margin,
            boxes[:, 1] >= boundary_margin,
            boxes[:, 2] <= w - boundary_margin,
            boxes[:, 3] <= h - boundary_margin
        ])
        
        box_widths = boxes[:, 2] - boxes[:, 0]
        box_heights = boxes[:, 3] - boxes[:, 1]
        box_areas = box_widths * box_heights
        
        min_area = (h * w) * 0.0005
        max_area = (h * w) * 0.5
        size_filter = (box_areas >= min_area) & (box_areas <= max_area)
        
        stability_filter = self._check_detection_stability(boxes)
        
        combined_filter = np.logical_and.reduce([boundary_filter, size_filter, stability_filter])
        
        return combined_filter.astype(bool)
    
    def _check_detection_stability(self, boxes: np.ndarray) -> np.ndarray:
        if len(boxes) == 0 or not self.previous_detections:
            return np.ones(len(boxes), dtype=bool)
        
        stability_mask = np.ones(len(boxes), dtype=bool)
        
        for i, box in enumerate(boxes):
            best_iou = 0
            best_prev_box = None
            
            for prev_box in self.previous_detections.get('boxes', []):
                iou = self._calculate_single_iou(box, prev_box)
                if iou > best_iou:
                    best_iou = iou
                    best_prev_box = prev_box
            
            if best_iou > 0.3 and best_prev_box is not None:
                current_area = (box[2] - box[0]) * (box[3] - box[1])
                prev_area = (best_prev_box[2] - best_prev_box[0]) * (best_prev_box[3] - best_prev_box[1])
                
                if prev_area > 0:
                    size_change = abs(current_area - prev_area) / prev_area
                    
                    if size_change > self.size_change_threshold:
                        stability_mask[i] = False
                        self.logger.debug(f"Rejected unstable detection with {size_change:.2f} size change")
        
        return stability_mask
    
    def _apply_confidence_boosting(self, boxes: np.ndarray, scores: np.ndarray) -> np.ndarray:
        if len(boxes) == 0:
            return scores
        
        boosted_scores = scores.copy()
        
        for i, (box, score) in enumerate(zip(boxes, scores)):
            persistence_count = self._get_detection_persistence(box)
            
            if persistence_count > 1:
                boost = min(persistence_count * self.persistence_boost_factor, 
                           self.max_confidence_boost)
                boosted_scores[i] = min(1.0, score + boost)
                
                if boost > 0:
                    self.logger.debug(f"Boosted detection confidence from {score:.3f} to {boosted_scores[i]:.3f} "
                                    f"(persistence: {persistence_count})")
        
        return boosted_scores
    
    def _get_detection_persistence(self, box: np.ndarray) -> int:
        persistence = 1
        
        for frame_idx in range(max(0, self.frame_count - 5), self.frame_count):
            if frame_idx in self.detection_history:
                for hist_box in self.detection_history[frame_idx]:
                    if self._calculate_single_iou(box, hist_box) > 0.5:
                        persistence += 1
                        break
        
        return min(persistence, 5)
    
    def _update_detection_history(self, detection_data: Dict):
        self.previous_detections = {
            'boxes': detection_data['boxes'].copy() if len(detection_data['boxes']) > 0 else [],
            'scores': detection_data['scores'].copy() if len(detection_data['scores']) > 0 else []
        }
        
        self.detection_history[self.frame_count] = detection_data['boxes'].copy() if len(detection_data['boxes']) > 0 else []
        
        frames_to_keep = 10
        old_frames = [f for f in self.detection_history.keys() if f < self.frame_count - frames_to_keep]
        for frame in old_frames:
            del self.detection_history[frame]
    
    def _calculate_single_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)
    
    def _extract_features(self, boxes: np.ndarray, frame: np.ndarray) -> Optional[np.ndarray]:
        if len(boxes) == 0:
            return None
        
        return np.random.rand(len(boxes), 512)
    
    def _empty_detection(self) -> Dict:
        return {
            'boxes': np.array([]).reshape(0, 4),
            'scores': np.array([]),
            'features': None,
            'classes': np.array([]),
            'ball_boxes': np.array([]).reshape(0, 4),
            'ball_scores': np.array([]),
            'referee_boxes': np.array([]).reshape(0, 4),
            'referee_scores': np.array([]),
            'frame_shape': None,
            'timestamp': self.frame_count,
            'total_detections': 0,
            'player_detections': 0
        }
    
    def get_stats(self) -> Dict:
        avg_detections = self.total_detections / max(self.frame_count, 1)
        return {
            'frames_processed': self.frame_count,
            'total_detections': self.total_detections,
            'avg_detections_per_frame': avg_detections,
            'model_path': self.config['model_path'],
            'class_names': self.class_names,
            'confidence_threshold': self.conf_threshold,
            'stability_rejections': getattr(self, 'stability_rejections', 0),
            'confidence_boosts_applied': getattr(self, 'confidence_boosts_applied', 0)
        }
