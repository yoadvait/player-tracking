import cv2
import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

@dataclass
class Track:
    track_id: int
    bbox: np.ndarray
    confidence: float
    features: Optional[np.ndarray]
    team_id: Optional[int] = None
    last_seen: int = 0
    hits: int = 1
    time_since_update: int = 0
    state: str = 'tentative'
    velocity: Optional[np.ndarray] = None
    predicted_bbox: Optional[np.ndarray] = None
    consecutive_misses: int = 0
    max_consecutive_misses: int = 12
    
    initial_confidence: Optional[float] = None
    confidence_decay_rate: float = 0.02
    min_confidence_threshold: float = 0.1

class PlayerTracker:
    
    def __init__(self, config: Dict):
        self.config = config['tracking']
        self.logger = logging.getLogger(__name__)
        
        self.max_age = self.config.get('track_buffer', 30)
        self.min_hits = 2
        self.track_high_thresh = self.config.get('track_high_thresh', 0.4)
        self.track_low_thresh = self.config.get('track_low_thresh', 0.1)
        
        self.iou_threshold = 0.5
        self.similarity_threshold = self.config.get('appearance_thresh', 0.3)
        self.proximity_threshold = self.config.get('proximity_thresh', 0.5)
        
        self.high_conf_iou_threshold = 0.6
        self.low_conf_iou_threshold = 0.4
        self.conf_threshold_boundary = 0.6
        
        self.max_size_change_ratio = 0.5
        
        self.expected_feature_dim = None
        
        self.occlusion_threshold = 0.7
        self.max_lost_frames = 15
        
        self.tracks = []
        self.lost_tracks = []
        self.track_history = defaultdict(lambda: deque(maxlen=30))
        self.next_id = 1
        self.frame_count = 0
        
        self.total_tracks = 0
        self.active_tracks = 0
        self.id_switches = 0
        
    def update(self, detections: Dict, frame: np.ndarray) -> List[Track]:
        self.frame_count += 1
        
        if detections['boxes'] is None or len(detections['boxes']) == 0:
            return self._update_tracks_no_detections()
        
        detection_boxes = detections['boxes']
        detection_scores = detections['scores']
        detection_features = detections.get('features')
        
        if (detection_features is not None and len(detection_features) > 0 and 
            self.expected_feature_dim is None):
            self.expected_feature_dim = detection_features[0].shape[0]
            self.logger.info(f"✅ Expected feature dimension set to: {self.expected_feature_dim}")
        
        self._predict_tracks_with_motion()
        
        matched_tracks, unmatched_dets, unmatched_tracks = self._enhanced_association(
            detection_boxes, detection_scores, detection_features
        )
        
        for track_idx, det_idx in matched_tracks:
            self._update_track_enhanced(
                self.tracks[track_idx], 
                detection_boxes[det_idx], 
                detection_scores[det_idx],
                detection_features[det_idx] if detection_features is not None else None
            )
        
        self._handle_unmatched_tracks_with_decay(unmatched_tracks)
        
        for det_idx in unmatched_dets:
            if (det_idx < len(detection_scores) and 
                detection_scores[det_idx] >= self.track_high_thresh):
                self._create_track_enhanced(
                    detection_boxes[det_idx],
                    detection_scores[det_idx],
                    detection_features[det_idx] if detection_features is not None else None
                )
        
        self._manage_lost_tracks()
        
        self._cleanup_tracks()
        
        self._update_track_history()
        
        confirmed_tracks = [t for t in self.tracks if t.state == 'confirmed']
        self.active_tracks = len(confirmed_tracks)
        
        return confirmed_tracks
    
    def _predict_tracks_with_motion(self):
        for track in self.tracks:
            track.time_since_update += 1
            
            history = self.track_history[track.track_id]
            if len(history) >= 2:
                current_center = np.array([(track.bbox[0] + track.bbox[2])/2, 
                                         (track.bbox[1] + track.bbox[3])/2])
                prev_center = np.array(history[-1])
                track.velocity = current_center - prev_center
                
                if track.velocity is not None:
                    predicted_center = current_center + track.velocity
                    bbox_width = track.bbox[2] - track.bbox[0]
                    bbox_height = track.bbox[3] - track.bbox[1]
                    
                    track.predicted_bbox = np.array([
                        predicted_center[0] - bbox_width/2,
                        predicted_center[1] - bbox_height/2,
                        predicted_center[0] + bbox_width/2,
                        predicted_center[1] + bbox_height/2
                    ])
    
    def _enhanced_association(self, det_boxes: np.ndarray, det_scores: np.ndarray, 
                            det_features: Optional[np.ndarray]) -> Tuple[List, List, List]:
        
        if len(self.tracks) == 0 or len(det_boxes) == 0:
            return [], list(range(len(det_boxes))), list(range(len(self.tracks)))
        
        iou_matrix = self._calculate_iou_matrix(det_boxes)
        high_iou_matches, remaining_dets, remaining_tracks = self._match_by_iou_adaptive(
            iou_matrix, det_scores, det_boxes, threshold=self.iou_threshold
        )
        
        if remaining_tracks and remaining_dets:
            pred_matches, remaining_dets, remaining_tracks = self._match_by_prediction(
                det_boxes, remaining_dets, remaining_tracks
            )
            high_iou_matches.extend(pred_matches)
        
        if det_features is not None and remaining_tracks and remaining_dets:
            app_matches, remaining_dets, remaining_tracks = self._match_by_appearance(
                det_features, remaining_dets, remaining_tracks
            )
            high_iou_matches.extend(app_matches)
        
        return high_iou_matches, remaining_dets, remaining_tracks
    
    def _calculate_iou_matrix(self, det_boxes: np.ndarray) -> np.ndarray:
        if len(self.tracks) == 0:
            return np.array([])
        
        track_boxes = []
        for track in self.tracks:
            if track.predicted_bbox is not None:
                track_boxes.append(track.predicted_bbox)
            else:
                track_boxes.append(track.bbox)
        
        track_boxes = np.array(track_boxes)
        return self._iou_batch(det_boxes, track_boxes)
    
    def _match_by_iou_adaptive(self, iou_matrix: np.ndarray, det_scores: np.ndarray, 
                              det_boxes: np.ndarray, threshold: float) -> Tuple[List, List, List]:
        matches = []
        unmatched_dets = list(range(len(det_scores)))
        unmatched_tracks = list(range(len(self.tracks)))
        
        if iou_matrix.size == 0:
            return matches, unmatched_dets, unmatched_tracks
        
        while len(unmatched_dets) > 0 and len(unmatched_tracks) > 0:
            valid_costs = iou_matrix[np.ix_(unmatched_dets, unmatched_tracks)]
            if valid_costs.size == 0:
                break
                
            max_iou_idx = np.unravel_index(np.argmax(valid_costs), valid_costs.shape)
            max_iou = valid_costs[max_iou_idx]
            
            det_idx = unmatched_dets[max_iou_idx[0]]
            track_idx = unmatched_tracks[max_iou_idx[1]]
            
            det_confidence = det_scores[det_idx]
            if det_confidence >= self.conf_threshold_boundary:
                required_iou = self.high_conf_iou_threshold
            else:
                required_iou = self.low_conf_iou_threshold
            
            if max_iou >= required_iou:
                if self._check_size_consistency(det_boxes[det_idx], self.tracks[track_idx].bbox):
                    matches.append([track_idx, det_idx])
                    unmatched_dets.remove(det_idx)
                    unmatched_tracks.remove(track_idx)
                else:
                    iou_matrix[det_idx, track_idx] = 0
                    continue
            else:
                break
        
        return matches, unmatched_dets, unmatched_tracks
    
    def _check_size_consistency(self, det_bbox: np.ndarray, track_bbox: np.ndarray) -> bool:
        det_area = (det_bbox[2] - det_bbox[0]) * (det_bbox[3] - det_bbox[1])
        track_area = (track_bbox[2] - track_bbox[0]) * (track_bbox[3] - track_bbox[1])
        
        if track_area == 0:
            return True
        
        size_ratio = abs(det_area - track_area) / track_area
        
        return size_ratio <= self.max_size_change_ratio
    
    def _match_by_prediction(self, det_boxes: np.ndarray, remaining_dets: List, 
                           remaining_tracks: List) -> Tuple[List, List, List]:
        matches = []
        
        for track_idx in remaining_tracks[:]:
            track = self.tracks[track_idx]
            if track.predicted_bbox is None:
                continue
            
            best_match = None
            best_iou = 0.2
            
            for det_idx in remaining_dets:
                iou = self._calculate_single_iou(det_boxes[det_idx], track.predicted_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_match = det_idx
            
            if best_match is not None:
                matches.append([track_idx, best_match])
                remaining_dets.remove(best_match)
                remaining_tracks.remove(track_idx)
        
        return matches, remaining_dets, remaining_tracks
    
    def _match_by_appearance(self, det_features: np.ndarray, remaining_dets: List,
                           remaining_tracks: List) -> Tuple[List, List, List]:
        matches = []
        
        if len(remaining_dets) == 0 or len(remaining_tracks) == 0:
            return matches, remaining_dets, remaining_tracks
        
        valid_track_indices = []
        valid_track_features = []
        
        for i, track_idx in enumerate(remaining_tracks):
            track = self.tracks[track_idx]
            if (track.features is not None and 
                len(det_features) > 0 and
                track.features.shape[0] == det_features[0].shape[0]):
                valid_track_indices.append(i)
                valid_track_features.append(track.features)
        
        if len(valid_track_features) == 0:
            return matches, remaining_dets, remaining_tracks
        
        det_feats = det_features[remaining_dets]
        track_feats = np.array(valid_track_features)
        
        similarity_matrix = self._cosine_similarity(det_feats, track_feats)
        
        threshold = 0.3
        remaining_dets_copy = remaining_dets[:]
        remaining_tracks_copy = [remaining_tracks[i] for i in valid_track_indices]
        
        while len(remaining_dets_copy) > 0 and len(remaining_tracks_copy) > 0:
            if similarity_matrix.size == 0:
                break
                
            max_sim_idx = np.unravel_index(np.argmax(similarity_matrix), 
                                         similarity_matrix.shape)
            max_sim = similarity_matrix[max_sim_idx]
            
            if max_sim >= threshold:
                det_idx = remaining_dets_copy[max_sim_idx[0]]
                track_idx = remaining_tracks_copy[max_sim_idx[1]]
                
                matches.append([track_idx, det_idx])
                
                det_pos = remaining_dets_copy.index(det_idx)
                track_pos = remaining_tracks_copy.index(track_idx)
                
                remaining_dets.remove(det_idx)
                remaining_tracks.remove(track_idx)
                
                remaining_dets_copy.remove(det_idx)
                remaining_tracks_copy.remove(track_idx)
                
                similarity_matrix = np.delete(similarity_matrix, det_pos, axis=0)
                similarity_matrix = np.delete(similarity_matrix, track_pos, axis=1)
            else:
                break
        
        return matches, remaining_dets, remaining_tracks
    
    def _handle_unmatched_tracks_with_decay(self, unmatched_tracks: List):
        tracks_to_move_to_lost = []
        
        for track_idx in unmatched_tracks:
            if track_idx < len(self.tracks):
                track = self.tracks[track_idx]
                track.consecutive_misses += 1
                
                if track.initial_confidence is None:
                    track.initial_confidence = track.confidence
                
                decay_amount = track.confidence_decay_rate * track.consecutive_misses
                track.confidence = max(
                    track.initial_confidence - decay_amount,
                    track.min_confidence_threshold
                )
                
                if track.consecutive_misses <= track.max_consecutive_misses // 2:
                    track.state = 'tentative'
                elif track.consecutive_misses <= track.max_consecutive_misses:
                    if track.confidence > track.min_confidence_threshold:
                        track.state = 'lost'
                    else:
                        tracks_to_move_to_lost.append(track_idx)
                else:
                    tracks_to_move_to_lost.append(track_idx)
        
        for track_idx in reversed(sorted(tracks_to_move_to_lost)):
            if track_idx < len(self.tracks):
                lost_track = self.tracks.pop(track_idx)
                lost_track.state = 'lost'
                lost_track.last_seen = self.frame_count - lost_track.time_since_update
                self.lost_tracks.append(lost_track)
                self.logger.debug(f"Track {lost_track.track_id} moved to lost tracks after {lost_track.consecutive_misses} misses")
    
    def _manage_lost_tracks(self):
        current_time = self.frame_count
        
        self.lost_tracks = [
            track for track in self.lost_tracks
            if current_time - track.last_seen <= self.max_lost_frames
        ]
        
        if len(self.lost_tracks) > 0:
            self.logger.debug(f"Managing {len(self.lost_tracks)} lost tracks")
    
    def _update_track_enhanced(self, track: Track, bbox: np.ndarray, 
                             confidence: float, features: Optional[np.ndarray]):
        track.bbox = bbox.copy()
        track.confidence = confidence
        track.last_seen = self.frame_count
        track.hits += 1
        track.time_since_update = 0
        track.consecutive_misses = 0
        
        track.initial_confidence = confidence
        
        if features is not None:
            if track.features is not None:
                if track.features.shape[0] == features.shape[0]:
                    alpha = 0.8
                    track.features = alpha * track.features + (1 - alpha) * features
                else:
                    self.logger.warning(f"Feature dimension mismatch for track {track.track_id}: "
                                      f"{track.features.shape[0]} vs {features.shape[0]}. "
                                      f"Replacing with new features.")
                    track.features = features.copy()
            else:
                track.features = features.copy()
        
        if track.hits >= self.min_hits and track.state != 'confirmed':
            track.state = 'confirmed'
        elif track.state in ['tentative', 'lost']:
            track.state = 'confirmed'
    
    def _create_track_enhanced(self, bbox: np.ndarray, confidence: float, 
                             features: Optional[np.ndarray]):
        track = Track(
            track_id=self.next_id,
            bbox=bbox.copy(),
            confidence=confidence,
            features=features.copy() if features is not None else None,
            last_seen=self.frame_count
        )
        
        self.tracks.append(track)
        self.next_id += 1
        self.total_tracks += 1
    
    def _cleanup_tracks(self):
        self.tracks = [t for t in self.tracks if t.state != 'deleted']
    
    def _update_tracks_no_detections(self) -> List[Track]:
        unmatched_track_indices = list(range(len(self.tracks)))
        self._handle_unmatched_tracks_with_decay(unmatched_track_indices)
        self._manage_lost_tracks()
        self._cleanup_tracks()
        return [t for t in self.tracks if t.state == 'confirmed']
    
    def _update_track_history(self):
        for track in self.tracks:
            if track.state == 'confirmed':
                center_x = (track.bbox[0] + track.bbox[2]) / 2
                center_y = (track.bbox[1] + track.bbox[3]) / 2
                self.track_history[track.track_id].append((center_x, center_y))
    
    def _iou_batch(self, bb_test: np.ndarray, bb_gt: np.ndarray) -> np.ndarray:
        if len(bb_test) == 0 or len(bb_gt) == 0:
            return np.array([])
            
        bb_test = np.expand_dims(bb_test, 1)
        bb_gt = np.expand_dims(bb_gt, 0)
        
        xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
        yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
        xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
        yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
        
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        
        intersection = w * h
        area_test = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
        area_gt = (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
        union = area_test + area_gt - intersection
        
        return intersection / (union + 1e-6)
    
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
    
    def _cosine_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> np.ndarray:
        norm1 = np.linalg.norm(feat1, axis=1, keepdims=True) + 1e-6
        norm2 = np.linalg.norm(feat2, axis=1, keepdims=True) + 1e-6
        
        return np.dot(feat1 / norm1, (feat2 / norm2).T)
    
    def get_track_history(self, track_id: int) -> List[Tuple[float, float]]:
        return list(self.track_history[track_id])
    
    def get_stats(self) -> Dict:
        avg_track_length = 0
        if self.track_history:
            lengths = [len(hist) for hist in self.track_history.values()]
            avg_track_length = np.mean(lengths) if lengths else 0
            
        return {
            'frames_processed': self.frame_count,
            'total_tracks': self.total_tracks,
            'active_tracks': self.active_tracks,
            'lost_tracks': len(self.lost_tracks),
            'id_switches': self.id_switches,
            'avg_track_length': avg_track_length,
            'expected_feature_dim': self.expected_feature_dim
        }
