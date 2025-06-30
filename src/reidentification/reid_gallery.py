import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
import pickle
from pathlib import Path

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

@dataclass
class GalleryEntry:
    track_id: int
    features: np.ndarray
    timestamp: int
    confidence: float
    bbox: np.ndarray
    team_id: Optional[int] = None
    jersey_number: Optional[str] = None
    pose_features: Optional[np.ndarray] = None
    color_features: Optional[np.ndarray] = None
    
    last_seen: int = 0
    hit_count: int = 1
    quality_score: float = 1.0
    
class ReIDGallery:
    
    def __init__(self, config: Dict):
        self.config = config['advanced_reid']['gallery']
        self.logger = logging.getLogger(__name__)
        
        self.max_size = self.config.get('max_size', 200)
        self.similarity_threshold = self.config.get('similarity_threshold', 0.6)
        self.temporal_decay = self.config.get('temporal_decay', 0.95)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        
        self.gallery: List[GalleryEntry] = []
        self.feature_index = None
        self.feature_dim = None
        self.use_faiss = FAISS_AVAILABLE
        
        self.active_tracks = set()
        self.inactive_tracks = set()
        self.track_history = defaultdict(list)
        
        self.total_entries = 0
        self.successful_reids = 0
        self.false_matches = 0
        
        if not self.use_faiss:
            self.logger.warning("⚠️  FAISS not available, using brute force search")
        
        self.logger.info("✅ ReID Gallery initialized")
    
    def add_entry(self, track_id: int, features: np.ndarray, timestamp: int,
                  confidence: float, bbox: np.ndarray, **kwargs):
        
        if features is None or len(features) == 0:
            self.logger.warning(f"Invalid features for track {track_id}")
            return
        
        existing_entry = self._find_track_entry(track_id)
        
        if existing_entry:
            self._update_entry(existing_entry, features, timestamp, confidence, bbox, **kwargs)
        else:
            entry = GalleryEntry(
                track_id=track_id,
                features=features.copy(),
                timestamp=timestamp,
                confidence=confidence,
                bbox=bbox.copy(),
                **kwargs
            )
            
            self.gallery.append(entry)
            self.total_entries += 1
            
            self._safe_update_index()
        
        self.active_tracks.add(track_id)
        self.inactive_tracks.discard(track_id)
        
        if len(self.gallery) > self.max_size:
            self._cleanup_gallery()
    
    def query_gallery(self, features: np.ndarray, exclude_tracks: Optional[List[int]] = None,
                     top_k: int = 5) -> List[Tuple[int, float, GalleryEntry]]:
        if len(self.gallery) == 0:
            return []
        
        exclude_tracks = exclude_tracks or []
        
        try:
            if self.use_faiss and self.feature_index is not None:
                return self._safe_faiss_search(features, exclude_tracks, top_k)
            else:
                return self._brute_force_search(features, exclude_tracks, top_k)
                
        except Exception as e:
            self.logger.warning(f"Gallery query failed, using brute force: {e}")
            return self._brute_force_search(features, exclude_tracks, top_k)
    
    def find_matches(self, features: np.ndarray, current_tracks: List[int],
                    timestamp: int) -> List[Tuple[int, float]]:
        self._apply_temporal_decay(timestamp)
        
        candidates = self.query_gallery(features, exclude_tracks=current_tracks, top_k=10)
        
        matches = []
        for track_id, similarity, entry in candidates:
            if similarity >= self.similarity_threshold:
                if self._verify_match(features, entry, timestamp):
                    confidence = self._calculate_match_confidence(similarity, entry, timestamp)
                    matches.append((track_id, confidence))
        
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches
    
    def mark_track_inactive(self, track_id: int, timestamp: int):
        self.active_tracks.discard(track_id)
        self.inactive_tracks.add(track_id)
        
        entry = self._find_track_entry(track_id)
        if entry:
            entry.last_seen = timestamp
    
    def reactivate_track(self, old_track_id: int, new_track_id: int, timestamp: int):
        old_entry = self._find_track_entry(old_track_id)
        if old_entry:
            new_entry = GalleryEntry(
                track_id=new_track_id,
                features=old_entry.features.copy(),
                timestamp=timestamp,
                confidence=old_entry.confidence,
                bbox=old_entry.bbox.copy(),
                team_id=old_entry.team_id,
                jersey_number=old_entry.jersey_number,
                pose_features=old_entry.pose_features,
                color_features=old_entry.color_features,
                last_seen=timestamp,
                hit_count=old_entry.hit_count + 1,
                quality_score=old_entry.quality_score
            )
            
            self.gallery.append(new_entry)
            self.successful_reids += 1
            
            self.inactive_tracks.discard(old_track_id)
            self.active_tracks.add(new_track_id)
            
            self._safe_update_index()
            
            self.logger.info(f"🎯 Successful re-identification: {old_track_id} -> {new_track_id}")
    
    def _find_track_entry(self, track_id: int) -> Optional[GalleryEntry]:
        for entry in self.gallery:
            if entry.track_id == track_id:
                return entry
        return None
    
    def _update_entry(self, entry: GalleryEntry, features: np.ndarray, 
                     timestamp: int, confidence: float, bbox: np.ndarray, **kwargs):
        if entry.features.shape[0] == features.shape[0]:
            alpha = 0.7
            entry.features = alpha * entry.features + (1 - alpha) * features
        else:
            self.logger.warning(f"Feature dimension mismatch in gallery entry {entry.track_id}: "
                              f"{entry.features.shape[0]} vs {features.shape[0]}. Replacing.")
            entry.features = features.copy()
        
        entry.timestamp = timestamp
        entry.confidence = max(entry.confidence, confidence)
        entry.bbox = bbox.copy()
        entry.hit_count += 1
        entry.last_seen = timestamp
        
        for key, value in kwargs.items():
            if hasattr(entry, key) and value is not None:
                setattr(entry, key, value)
        
        entry.quality_score = self._calculate_quality_score(entry)
    
    def _safe_update_index(self):
        if not self.use_faiss or len(self.gallery) == 0:
            return
        
        try:
            features_list = []
            for entry in self.gallery:
                if entry.features is not None and len(entry.features) > 0:
                    features_list.append(entry.features)
            
            if not features_list:
                return
            
            feature_dims = [f.shape[0] for f in features_list]
            if len(set(feature_dims)) > 1:
                self.logger.warning(f"Inconsistent feature dimensions: {set(feature_dims)}. "
                                  f"Disabling FAISS index.")
                self.use_faiss = False
                return
            
            features = np.array(features_list)
            
            if self.feature_dim is None:
                self.feature_dim = features.shape[1]
            
            self.feature_index = faiss.IndexFlatIP(self.feature_dim)
            
            features_normalized = features.copy().astype(np.float32)
            faiss.normalize_L2(features_normalized)
            self.feature_index.add(features_normalized)
            
        except Exception as e:
            self.logger.warning(f"Failed to update FAISS index: {e}. Using brute force search.")
            self.use_faiss = False
            self.feature_index = None
    
    def _safe_faiss_search(self, query_features: np.ndarray, exclude_tracks: List[int], 
                          k: int) -> List[Tuple[int, float, GalleryEntry]]:
        try:
            query = query_features.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(query)
            
            similarities, indices = self.feature_index.search(query, min(k * 2, len(self.gallery)))
            
            results = []
            for sim, idx in zip(similarities[0], indices[0]):
                if idx < len(self.gallery):
                    entry = self.gallery[idx]
                    if entry.track_id not in exclude_tracks:
                        results.append((entry.track_id, float(sim), entry))
            
            return results[:k]
            
        except Exception as e:
            self.logger.warning(f"FAISS search failed: {e}. Falling back to brute force.")
            self.use_faiss = False
            return self._brute_force_search(query_features, exclude_tracks, k)
    
    def _brute_force_search(self, features: np.ndarray, exclude_tracks: List[int], 
                           top_k: int) -> List[Tuple[int, float, GalleryEntry]]:
        similarities = []
        
        for entry in self.gallery:
            if entry.track_id not in exclude_tracks and entry.features is not None:
                try:
                    if entry.features.shape[0] == features.shape[0]:
                        sim = np.dot(features, entry.features) / (
                            np.linalg.norm(features) * np.linalg.norm(entry.features) + 1e-8
                        )
                        similarities.append((entry.track_id, sim, entry))
                except Exception as e:
                    self.logger.warning(f"Similarity calculation failed for track {entry.track_id}: {e}")
                    continue
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def _verify_match(self, features: np.ndarray, entry: GalleryEntry, timestamp: int) -> bool:
        
        time_diff = timestamp - entry.last_seen
        if time_diff > 300:  
            return False
        
        if entry.quality_score < 0.3:
            return False
        
        return True
    
    def _calculate_match_confidence(self, similarity: float, entry: GalleryEntry, 
                                  timestamp: int) -> float:
        
        confidence = similarity
        
        confidence *= (0.5 + 0.5 * entry.quality_score)
        
        hit_boost = min(0.2, entry.hit_count * 0.02)
        confidence += hit_boost
        
        time_diff = timestamp - entry.last_seen
        time_penalty = min(0.3, time_diff * 0.001)
        confidence -= time_penalty
        
        return np.clip(confidence, 0.0, 1.0)
    
    def _calculate_quality_score(self, entry: GalleryEntry) -> float:
        
        quality = entry.confidence
        
        hit_boost = min(0.3, entry.hit_count * 0.05)
        quality += hit_boost
        
        if entry.jersey_number:
            quality += 0.1
        if entry.team_id is not None:
            quality += 0.05
        
        return np.clip(quality, 0.0, 1.0)
    
    def _apply_temporal_decay(self, current_timestamp: int):
        for entry in self.gallery:
            time_diff = current_timestamp - entry.timestamp
            decay_factor = self.temporal_decay ** (time_diff / 30.0)
            entry.quality_score *= decay_factor
    
    def _cleanup_gallery(self):
        self.gallery.sort(key=lambda x: x.quality_score)
        entries_to_remove = len(self.gallery) - self.max_size
        removed_entries = self.gallery[:entries_to_remove]
        self.gallery = self.gallery[entries_to_remove:]
        
        for entry in removed_entries:
            self.inactive_tracks.discard(entry.track_id)
        
        self._safe_update_index()
        
        self.logger.info(f"🧹 Cleaned up gallery: removed {entries_to_remove} entries")
    
    def save_gallery(self, path: str):
        try:
            gallery_data = {
                'entries': self.gallery,
                'active_tracks': self.active_tracks,
                'inactive_tracks': self.inactive_tracks,
                'statistics': self.get_statistics()
            }
            
            with open(path, 'wb') as f:
                pickle.dump(gallery_data, f)
                
            self.logger.info(f"💾 Gallery saved to {path}")
        except Exception as e:
            self.logger.error(f"Failed to save gallery: {e}")
    
    def load_gallery(self, path: str):
        try:
            with open(path, 'rb') as f:
                gallery_data = pickle.load(f)
            
            self.gallery = gallery_data['entries']
            self.active_tracks = gallery_data['active_tracks']
            self.inactive_tracks = gallery_data['inactive_tracks']
            
            self._safe_update_index()
            
            self.logger.info(f"📁 Gallery loaded from {path}")
        except Exception as e:
            self.logger.error(f"Failed to load gallery: {e}")
    
    def get_statistics(self) -> Dict:
        return {
            'total_entries': len(self.gallery),
            'active_tracks': len(self.active_tracks),
            'inactive_tracks': len(self.inactive_tracks),
            'successful_reids': self.successful_reids,
            'false_matches': self.false_matches,
            'average_quality': np.mean([e.quality_score for e in self.gallery]) if self.gallery else 0,
            'feature_dimension': self.feature_dim,
            'using_faiss': self.use_faiss
        }
