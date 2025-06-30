import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

class AdvancedVisualizer:
    
    def __init__(self, config: Dict):
        self.config = config['visualization']
        self.logger = logging.getLogger(__name__)

        self.show_ids = self.config.get('show_ids', True)
        self.show_confidence = self.config.get('show_confidence', True)
        self.show_team_colors = self.config.get('show_team_colors', True)
        self.show_reid_info = self.config.get('show_reid_info', True)
        self.show_jersey_numbers = self.config.get('show_jersey_numbers', True)
        self.show_pose_keypoints = self.config.get('show_pose_keypoints', False)
        
        self.bbox_thickness = self.config.get('bbox_thickness', 2)
        self.font_scale = self.config.get('font_scale', 0.6)
        
        self.team_colors = {
            0: (255, 100, 100),
            1: (100, 100, 255),
            -1: (100, 255, 100),
            'reid': (255, 255, 100),
            'new': (200, 200, 200),
            'lost': (150, 150, 150)
        }
        
        self.frame_count = 0
        
    def draw_advanced_tracks(self, frame: np.ndarray, tracks: List, 
                           team_assignments: Dict[int, int] = None,
                           reid_info: Dict = None,
                           tracker_stats: Dict = None) -> np.ndarray:
        
        annotated_frame = frame.copy()
        
        for track in tracks:
            track_id = track.track_id
            bbox = track.bbox.astype(int)
            confidence = track.confidence
            
            track_status, color = self._get_track_status_and_color(
                track, team_assignments, reid_info
            )
            
            self._draw_enhanced_bbox(
                annotated_frame, bbox, color, track, track_status, 
                team_assignments, reid_info
            )
            
            if self.show_pose_keypoints and hasattr(track, 'deep_features'):
                self._draw_pose_keypoints(annotated_frame, bbox, track)
        
        self._draw_advanced_stats(annotated_frame, tracks, tracker_stats, reid_info)
        
        if self.show_reid_info and reid_info:
            self._draw_reid_panel(annotated_frame, reid_info)
        
        self.frame_count += 1
        return annotated_frame
    
    def _get_track_status_and_color(self, track, team_assignments: Dict, 
                                   reid_info: Dict) -> Tuple[str, Tuple[int, int, int]]:
        
        track_id = track.track_id
        
        if reid_info and track_id in reid_info.get('reidentified_tracks', []):
            return 'reid', self.team_colors['reid']
        
        if track.hits <= 3:
            return 'new', self.team_colors['new']
        
        if team_assignments and track_id in team_assignments:
            team_id = team_assignments[track_id]
            return f'team_{team_id}', self.team_colors.get(team_id, self.team_colors[-1])
        
        return 'unassigned', self.team_colors[-1]
    
    def _draw_enhanced_bbox(self, frame: np.ndarray, bbox: np.ndarray, 
                          color: Tuple[int, int, int], track, track_status: str,
                          team_assignments: Dict, reid_info: Dict):
        
        x1, y1, x2, y2 = bbox
        track_id = track.track_id
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.bbox_thickness)
        
        label_parts = []
        
        if self.show_ids:
            label_parts.append(f"ID:{track_id}")
        
        if (self.show_jersey_numbers and 
            hasattr(track, 'jersey_number') and track.jersey_number):
            label_parts.append(f"#{track.jersey_number}")
        
        if self.show_team_colors and team_assignments:
            team_id = team_assignments.get(track_id, -1)
            if team_id >= 0:
                label_parts.append(f"T{team_id}")
        
        if self.show_reid_info and reid_info:
            if track_id in reid_info.get('reidentified_tracks', []):
                label_parts.append("REID")
            elif track.hits <= 3:
                label_parts.append("NEW")
        
        if self.show_confidence:
            label_parts.append(f"{track.confidence:.2f}")
        
        if label_parts:
            self._draw_multi_line_label(frame, (x1, y1), label_parts, color)
        
        self._draw_track_indicators(frame, bbox, track, color)
    
    def _draw_multi_line_label(self, frame: np.ndarray, position: Tuple[int, int], 
                             label_parts: List[str], color: Tuple[int, int, int]):
        
        x, y = position
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = self.font_scale
        thickness = 1
        
        max_parts_per_line = 3
        lines = []
        for i in range(0, len(label_parts), max_parts_per_line):
            line = " | ".join(label_parts[i:i + max_parts_per_line])
            lines.append(line)
        
        line_height = 20
        max_width = 0
        for line in lines:
            text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
            max_width = max(max_width, text_size[0])
        
        bg_height = len(lines) * line_height + 10
        label_y = max(y - bg_height - 5, bg_height)
        
        cv2.rectangle(frame, 
                     (x, label_y - bg_height), 
                     (x + max_width + 10, label_y + 5), 
                     color, -1)
        
        for i, line in enumerate(lines):
            text_y = label_y - bg_height + 15 + i * line_height
            cv2.putText(frame, line, (x + 5, text_y), 
                       font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    def _draw_track_indicators(self, frame: np.ndarray, bbox: np.ndarray, 
                             track, color: Tuple[int, int, int]):
        
        x1, y1, x2, y2 = bbox
        
        quality = getattr(track, 'quality_score', track.confidence)
        quality_color = self._get_quality_color(quality)
        quality_center = (x2 - 8, y1 + 8)
        cv2.circle(frame, quality_center, 6, quality_color, -1)
        cv2.circle(frame, quality_center, 6, (255, 255, 255), 1)
        
        age_ratio = min(1.0, track.hits / 30.0)
        bar_width = int((x2 - x1) * age_ratio)
        if bar_width > 0:
            cv2.rectangle(frame, (x1, y2 - 3), (x1 + bar_width, y2), color, -1)
    
    def _get_quality_color(self, quality: float) -> Tuple[int, int, int]:
        if quality >= 0.8:
            return (0, 255, 0)
        elif quality >= 0.6:
            return (0, 255, 255)
        elif quality >= 0.4:
            return (0, 165, 255)
        else:
            return (0, 0, 255)
    
    def _draw_pose_keypoints(self, frame: np.ndarray, bbox: np.ndarray, track):
        """Draw pose keypoints if available"""
        
        if not hasattr(track, 'deep_features') or not track.deep_features:
            return
        
        pose_features = track.deep_features.get('pose')
        if pose_features is None or len(pose_features) < 99:
            return
        
        x1, y1, x2, y2 = bbox
        
        for i in range(0, len(pose_features), 3):
            if i + 2 < len(pose_features):
                rel_x, rel_y, visibility = pose_features[i:i+3]
                
                if visibility > 0.5:
                    abs_x = int(x1 + rel_x * (x2 - x1))
                    abs_y = int(y1 + rel_y * (y2 - y1))
                    
                    cv2.circle(frame, (abs_x, abs_y), 2, (0, 255, 255), -1)
    
    def _draw_advanced_stats(self, frame: np.ndarray, tracks: List, 
                           tracker_stats: Dict, reid_info: Dict):
        
        h, w = frame.shape[:2]
        
        total_players = len(tracks)
        active_tracks = len([t for t in tracks if t.state == 'confirmed'])
        new_tracks = len([t for t in tracks if t.hits <= 3])
        
        reid_stats = tracker_stats.get('reid_stats', {}) if tracker_stats else {}
        successful_reids = reid_stats.get('successful_reids', 0)
        reid_attempts = reid_stats.get('reid_attempts', 0)
        reid_rate = reid_stats.get('reid_success_rate', 0.0)
        
        jersey_detections = reid_stats.get('jersey_detections', 0)
        
        stats_lines = [
            f"Frame: {self.frame_count}",
            f"Players: {total_players} (Active: {active_tracks})",
            f"New Tracks: {new_tracks}",
            f"Re-IDs: {successful_reids}/{reid_attempts} ({reid_rate:.1%})",
            f"Jersey #s: {jersey_detections}"
        ]
        
        self._draw_stats_panel(frame, stats_lines, (10, 10))
    
    def _draw_reid_panel(self, frame: np.ndarray, reid_info: Dict):
        
        h, w = frame.shape[:2]
        
        recent_reids = reid_info.get('recent_reidentifications', [])
        
        if recent_reids:
            reid_lines = ["Recent Re-IDs:"]
            for reid_event in recent_reids[-3:]:
                old_id = reid_event.get('old_id', '?')
                new_id = reid_event.get('new_id', '?')
                confidence = reid_event.get('confidence', 0.0)
                reid_lines.append(f"  {old_id} -> {new_id} ({confidence:.2f})")
            
            panel_x = w - 250
            self._draw_stats_panel(frame, reid_lines, (panel_x, 10), 
                                 bg_color=(40, 40, 100))
    
    def _draw_stats_panel(self, frame: np.ndarray, lines: List[str], 
                         position: Tuple[int, int], 
                         bg_color: Tuple[int, int, int] = (0, 0, 0)):
        
        x, y = position
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        line_height = 25
        
        max_text_width = 0
        for line in lines:
            text_size = cv2.getTextSize(line, font, font_scale, 1)[0]
            max_text_width = max(max_text_width, text_size[0])
        
        panel_width = max_text_width + 20
        panel_height = len(lines) * line_height + 15
        
        cv2.rectangle(frame, (x, y), (x + panel_width, y + panel_height), bg_color, -1)
        cv2.rectangle(frame, (x, y), (x + panel_width, y + panel_height), (255, 255, 255), 2)
        
        for i, line in enumerate(lines):
            text_y = y + 20 + i * line_height
            cv2.putText(frame, line, (x + 10, text_y), 
                       font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    
    def create_reid_summary(self, reid_stats: Dict) -> np.ndarray:
        
        canvas = np.zeros((400, 600, 3), dtype=np.uint8)
        
        cv2.putText(canvas, "Re-Identification Summary", (150, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        stats_text = [
            f"Total Re-ID Attempts: {reid_stats.get('reid_attempts', 0)}",
            f"Successful Re-IDs: {reid_stats.get('successful_reids', 0)}",
            f"Success Rate: {reid_stats.get('reid_success_rate', 0.0):.1%}",
            f"Jersey Detections: {reid_stats.get('jersey_detections', 0)}",
            f"Gallery Entries: {reid_stats.get('gallery_stats', {}).get('total_entries', 0)}"
        ]
        
        for i, text in enumerate(stats_text):
            y_pos = 80 + i * 30
            cv2.putText(canvas, text, (50, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return canvas
