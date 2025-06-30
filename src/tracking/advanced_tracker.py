import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from collections import defaultdict, deque
from dataclasses import dataclass

from ..features.deep_extractor import DeepFeatureExtractor
from ..reidentification.reid_gallery import ReIDGallery
from ..features.jersey_ocr import JerseyNumberDetector
from .tracker import Track, PlayerTracker

class AdvancedPlayerTracker(PlayerTracker):
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.feature_extractor = DeepFeatureExtractor(config)
        self.reid_gallery = ReIDGallery(config)
        self.jersey_detector = JerseyNumberDetector(config)
        
        self.reid_enabled = config.get('advanced_reid', {}).get('enabled', True)
        self.reid_threshold = config.get('advanced_reid', {}).get('gallery', {}).get('similarity_threshold', 0.6)
        
        self.lost_track_buffer = 60
        self.reid_check_interval = 5
        
        self.lost_tracks = []
        self.reid_candidates = []
        
        self.reid_attempts = 0
        self.successful_reids = 0
        self.jersey_detections = 0
        
        self.logger.info("✅ Advanced tracker with re-identification initialized")
    
    def update(self, detections: Dict, frame: np.ndarray) -> List[Track]:
        
        self.frame_count += 1
        
        active_tracks = super().update(detections, frame)
        
        if self.reid_enabled:
            self._update_track_features(active_tracks, frame)
            
            if self.frame_count % self.reid_check_interval == 0:
                self._check_for_reidentification(active_tracks, frame)
        
        self._update_gallery(active_tracks)
        
        self._manage_lost_tracks()
        
        return active_tracks
    
    def _update_track_features(self, tracks: List[Track], frame: np.ndarray):
        
        for track in tracks:
            try:
                features_dict = self.feature_extractor.extract_features(frame, track.bbox, track.track_id)
                
                track.features = features_dict.get('deep', None)
                
                track.deep_features = features_dict
                
                if self.jersey_detector.enabled:
                    x1, y1, x2, y2 = track.bbox.astype(int)
                    player_crop = frame[y1:y2, x1:x2]
                    
                    jersey_result = self.jersey_detector.detect_number(player_crop)
                    if jersey_result:
                        track.jersey_number = jersey_result['number']
                        track.jersey_confidence = jersey_result['confidence']
                        self.jersey_detections += 1
                        
                        self.logger.debug(f"🔢 Jersey number detected: {jersey_result['number']} "
                                        f"(confidence: {jersey_result['confidence']:.2f}) for track {track.track_id}")
                
            except Exception as e:
                self.logger.warning(f"Feature extraction failed for track {track.track_id}: {e}")
    
    def _check_for_reidentification(self, active_tracks: List[Track], frame: np.ndarray):
        
        if len(self.lost_tracks) == 0:
            return
        
        active_track_ids = [t.track_id for t in active_tracks]
        
        new_tracks = [t for t in active_tracks if t.hits <= 3]
        
        for new_track in new_tracks:
            if new_track.features is None:
                continue
            
            self.reid_attempts += 1
            
            matches = self.reid_gallery.find_matches(
                new_track.features, 
                active_track_ids,
                self.frame_count
            )
            
            if matches:
                best_match_id, confidence = matches[0]
                
                if self._verify_reidentification(new_track, best_match_id, confidence):
                    self._perform_reidentification(new_track, best_match_id)
                    self.successful_reids += 1
                    
                    self.logger.info(f"🎯 Re-identification successful: "
                                   f"Track {new_track.track_id} -> {best_match_id} "
                                   f"(confidence: {confidence:.3f})")
    
    def _verify_reidentification(self, new_track: Track, candidate_id: int, confidence: float) -> bool:
        
        if confidence < self.reid_threshold:
            return False
        
        lost_track = None
        for track in self.lost_tracks:
            if track.track_id == candidate_id:
                lost_track = track
                break
        
        if lost_track is None:
            return False
        
        if (hasattr(new_track, 'jersey_number') and 
            hasattr(lost_track, 'jersey_number') and
            new_track.jersey_number and lost_track.jersey_number):
            
            jersey_similarity = self.jersey_detector.compare_numbers(
                new_track.jersey_number, lost_track.jersey_number
            )
            
            if jersey_similarity < 0.5:
                self.logger.debug(f"❌ Jersey number mismatch: "
                                f"{new_track.jersey_number} vs {lost_track.jersey_number}")
                return False
            else:
                self.logger.debug(f"✅ Jersey number match: "
                                f"{new_track.jersey_number} = {lost_track.jersey_number}")
        
        if (hasattr(new_track, 'team_id') and 
            hasattr(lost_track, 'team_id') and
            new_track.team_id != lost_track.team_id and
            new_track.team_id is not None and lost_track.team_id is not None):
            
            self.logger.debug(f"❌ Team mismatch: {new_track.team_id} vs {lost_track.team_id}")
            return False
        
        time_since_lost = self.frame_count - lost_track.last_seen
        if time_since_lost > self.lost_track_buffer:
            return False
        
        return True
    
    def _perform_reidentification(self, new_track: Track, old_track_id: int):
        
        old_track = None
        for i, track in enumerate(self.lost_tracks):
            if track.track_id == old_track_id:
                old_track = self.lost_tracks.pop(i)
                break
        
        if old_track is None:
            return
        
        old_id = new_track.track_id
        new_track.track_id = old_track_id
        
        if old_track_id in self.track_history:
            self.track_history[old_track_id].extend(self.track_history.get(old_id, []))
            if old_id in self.track_history:
                del self.track_history[old_id]
        
        if hasattr(old_track, 'team_id') and old_track.team_id is not None:
            new_track.team_id = old_track.team_id
        
        if hasattr(old_track, 'jersey_number') and old_track.jersey_number:
            if not hasattr(new_track, 'jersey_number') or not new_track.jersey_number:
                new_track.jersey_number = old_track.jersey_number
        
        self.reid_gallery.reactivate_track(old_track_id, old_track_id, self.frame_count)
    
    def _update_gallery(self, tracks: List[Track]):
        
        for track in tracks:
            if track.features is not None and track.state == 'confirmed':
                kwargs = {}
                if hasattr(track, 'team_id'):
                    kwargs['team_id'] = track.team_id
                if hasattr(track, 'jersey_number'):
                    kwargs['jersey_number'] = getattr(track, 'jersey_number', None)
                if hasattr(track, 'deep_features'):
                    kwargs['pose_features'] = track.deep_features.get('pose')
                    kwargs['color_features'] = track.deep_features.get('color')
                
                self.reid_gallery.add_entry(
                    track_id=track.track_id,
                    features=track.features,
                    timestamp=self.frame_count,
                    confidence=track.confidence,
                    bbox=track.bbox,
                    **kwargs
                )
    
    def _manage_lost_tracks(self):
        
        current_time = self.frame_count
        
        tracks_to_move = []
        for i, track in enumerate(self.tracks):
            if (track.time_since_update > 10 and 
                track.state == 'confirmed' and  
                track.features is not None):    
                
                tracks_to_move.append(i)
        
        for i in reversed(tracks_to_move):
            lost_track = self.tracks.pop(i)
            lost_track.last_seen = current_time - lost_track.time_since_update
            self.lost_tracks.append(lost_track)
            
            self.reid_gallery.mark_track_inactive(lost_track.track_id, lost_track.last_seen)
            
            self.logger.debug(f"📤 Track {lost_track.track_id} moved to lost tracks")
        
        self.lost_tracks = [
            track for track in self.lost_tracks 
            if current_time - track.last_seen <= self.lost_track_buffer
        ]
    
    def get_reid_statistics(self) -> Dict:
        
        base_stats = super().get_stats()
        
        reid_stats = {
            'reid_attempts': self.reid_attempts,
            'successful_reids': self.successful_reids,
            'reid_success_rate': self.successful_reids / max(self.reid_attempts, 1),
            'jersey_detections': self.jersey_detections,
            'lost_tracks': len(self.lost_tracks),
            'gallery_stats': self.reid_gallery.get_statistics()
        }
        
        base_stats.update(reid_stats)
        return base_stats
    
    def save_reid_data(self, output_dir: str):
        
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.reid_gallery.save_gallery(str(output_path / "reid_gallery.pkl"))
        
        stats = self.get_reid_statistics()
        import json
        with open(output_path / "reid_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info(f"💾 Re-identification data saved to {output_path}")
