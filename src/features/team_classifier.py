import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter, defaultdict
import logging
from typing import Dict, List, Tuple, Optional

class TeamClassifier:
    
    def __init__(self, config: Dict):
        self.config = config['team_classification']
        self.logger = logging.getLogger(__name__)
        
        self.enabled = self.config.get('enabled', True)
        self.n_clusters = self.config.get('n_clusters', 2)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        
        self.color_regions = self.config.get('color_regions', ['jersey_top', 'shorts'])
        
        self.team_colors = {}
        self.team_assignments = {}
        
        self.kmeans = None
        self.color_history = defaultdict(list)
        
    def classify_teams(self, frame: np.ndarray, tracks: List) -> Dict[int, int]:

        if not self.enabled or len(tracks) < 2:
            return {}
        
        color_features = []
        track_ids = []
        
        for track in tracks:
            colors = self._extract_player_colors(frame, track.bbox)
            if colors is not None:
                color_features.append(colors)
                track_ids.append(track.track_id)
        
        if len(color_features) < 2:
            return {}
        
        team_assignments = self._cluster_players(color_features, track_ids)
        
        confident_assignments = {}
        for track_id, team_id in team_assignments.items():
            if self._calculate_assignment_confidence(track_id, team_id) >= self.confidence_threshold:
                confident_assignments[track_id] = team_id
        
        return confident_assignments
    
    def _extract_player_colors(self, frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:

        try:
            x1, y1, x2, y2 = bbox.astype(int)
            
            player_crop = frame[y1:y2, x1:x2]
            
            if player_crop.size == 0:
                return None
            
            colors = []
            
            if 'jersey_top' in self.color_regions:
                jersey_region = player_crop[:int(player_crop.shape[0] * 0.6), :]
                jersey_color = self._get_dominant_color(jersey_region)
                colors.extend(jersey_color)
            
            if 'shorts' in self.color_regions:
                shorts_start = int(player_crop.shape[0] * 0.4)
                shorts_end = int(player_crop.shape[0] * 0.8)
                shorts_region = player_crop[shorts_start:shorts_end, :]
                shorts_color = self._get_dominant_color(shorts_region)
                colors.extend(shorts_color)
            
            return np.array(colors) if colors else None
            
        except Exception as e:
            self.logger.warning(f"Color extraction failed: {e}")
            return None
    
    def _get_dominant_color(self, region: np.ndarray, n_colors: int = 3) -> List[float]:

        if region.size == 0:
            return [0, 0, 0]
        
        pixels = region.reshape(-1, 3)
        
        mask = np.all((pixels > 20) & (pixels < 235), axis=1)
        if np.sum(mask) < 10:
            mask = np.ones(len(pixels), dtype=bool)
        
        valid_pixels = pixels[mask]
        
        if len(valid_pixels) < n_colors:
            return np.mean(valid_pixels, axis=0).tolist() if len(valid_pixels) > 0 else [0, 0, 0]
        
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(valid_pixels)
        
        colors = kmeans.cluster_centers_
        
        labels = kmeans.labels_
        weights = [np.sum(labels == i) for i in range(n_colors)]
        
        weighted_color = np.average(colors, weights=weights, axis=0)
        return weighted_color.tolist()
    
    def _cluster_players(self, color_features: List[np.ndarray], track_ids: List[int]) -> Dict[int, int]:
        
        features_array = np.array(color_features)
        
        if self.kmeans is None or len(color_features) != len(self.team_assignments):
            self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        
        cluster_labels = self.kmeans.fit_predict(features_array)
        
        team_assignments = {}
        for i, track_id in enumerate(track_ids):
            team_assignments[track_id] = int(cluster_labels[i])
        
        for track_id, team_id in team_assignments.items():
            self.color_history[track_id].append(team_id)
            if len(self.color_history[track_id]) > 10:
                self.color_history[track_id].pop(0)
        
        return team_assignments
    
    def _calculate_assignment_confidence(self, track_id: int, team_id: int) -> float:

        if track_id not in self.color_history:
            return 0.5
        
        history = self.color_history[track_id]
        if len(history) == 0:
            return 0.5
        
        team_counts = Counter(history)
        confidence = team_counts[team_id] / len(history)
        
        return confidence
    
    def get_team_colors(self) -> Dict[int, np.ndarray]:

        if self.kmeans is None:
            return {}
        
        team_colors = {}
        for i in range(self.n_clusters):
            team_colors[i] = self.kmeans.cluster_centers_[i]
        
        return team_colors
    
    def get_stats(self) -> Dict:
        
        total_assignments = sum(len(hist) for hist in self.color_history.values())
        unique_players = len(self.color_history)
        
        return {
            'total_assignments': total_assignments,
            'unique_players_classified': unique_players,
            'team_colors': self.get_team_colors(),
            'classification_enabled': self.enabled
        }
