import cv2
import numpy as np
import easyocr
import re
from typing import Dict, List, Tuple, Optional
import logging

class JerseyNumberDetector:
    
    def __init__(self, config: Dict):
        self.config = config['advanced_reid']['jersey_ocr']
        self.logger = logging.getLogger(__name__)
        
        self.enabled = self.config.get('enabled', True)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.6)
        self.preprocessing = self.config.get('preprocessing', True)
        
        if self.enabled:
            try:
                self.reader = easyocr.Reader(['en'], gpu=False)
                self.logger.info("✅ Jersey OCR initialized")
            except Exception as e:
                self.logger.warning(f"⚠️  Failed to initialize OCR: {e}")
                self.enabled = False
        else:
            self.reader = None
    
    def detect_number(self, player_crop: np.ndarray) -> Optional[Dict]:
        if not self.enabled or self.reader is None:
            return None
        
        try:
            if self.preprocessing:
                processed_crop = self._preprocess_for_ocr(player_crop)
            else:
                processed_crop = player_crop
            
            results = self.reader.readtext(processed_crop)
            
            best_number = self._process_ocr_results(results, processed_crop.shape)
            
            return best_number
            
        except Exception as e:
            self.logger.warning(f"Jersey number detection failed: {e}")
            return None
    
    def _preprocess_for_ocr(self, crop: np.ndarray) -> np.ndarray:
        
        h, w = crop.shape[:2]
        
        y1 = int(h * 0.2)
        y2 = int(h * 0.7)
        torso_crop = crop[y1:y2, :]
        
        gray = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2GRAY)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        processed_images = []
        
        processed_images.append(enhanced)
        
        _, thresh1 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_images.append(thresh1)
        
        _, thresh2 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        processed_images.append(thresh2)
        
        edges = cv2.Canny(enhanced, 50, 150)
        processed_images.append(edges)
        
        return enhanced
    
    def _process_ocr_results(self, results: List, image_shape: Tuple) -> Optional[Dict]:
        
        if not results:
            return None
        
        valid_numbers = []
        
        for (bbox, text, confidence) in results:
            clean_text = self._clean_text(text)
            
            if self._is_valid_jersey_number(clean_text) and confidence >= self.confidence_threshold:
                position_score = self._calculate_position_score(bbox, image_shape)
                
                size_score = self._calculate_size_score(bbox)
                
                total_score = confidence * 0.5 + position_score * 0.3 + size_score * 0.2
                
                valid_numbers.append({
                    'number': clean_text,
                    'confidence': confidence,
                    'position_score': position_score,
                    'size_score': size_score,
                    'total_score': total_score,
                    'bbox': bbox
                })
        
        if valid_numbers:
            best_number = max(valid_numbers, key=lambda x: x['total_score'])
            return best_number
        
        return None
    
    def _clean_text(self, text: str) -> str:
        cleaned = re.sub(r'[^0-9]', '', text)
        
        replacements = {
            'O': '0', 'o': '0', 'I': '1', 'l': '1', 'S': '5', 's': '5',
            'G': '6', 'B': '8', 'Z': '2', 'T': '7'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        numbers = re.findall(r'\d+', text)
        
        if numbers:
            return max(numbers, key=len)
        
        return cleaned
    
    def _is_valid_jersey_number(self, text: str) -> bool:
        if not text or not text.isdigit():
            return False
        
        number = int(text)
        
        if 0 <= number <= 99:
            return True
        
        if len(text) <= 2:
            return True
        
        return False
    
    def _calculate_position_score(self, bbox: List, image_shape: Tuple) -> float:
        h, w = image_shape[:2]
        
        bbox_array = np.array(bbox)
        center_x = np.mean(bbox_array[:, 0])
        center_y = np.mean(bbox_array[:, 1])
        
        norm_x = center_x / w
        norm_y = center_y / h
        
        x_score = 1.0 - abs(norm_x - 0.5) * 2
        y_score = 1.0 - abs(norm_y - 0.4) * 2.5
        
        position_score = (x_score + y_score) / 2
        
        return max(0, position_score)
    
    def _calculate_size_score(self, bbox: List) -> float:
        bbox_array = np.array(bbox)
        
        width = np.max(bbox_array[:, 0]) - np.min(bbox_array[:, 0])
        height = np.max(bbox_array[:, 1]) - np.min(bbox_array[:, 1])
        area = width * height
        
        normalized_area = min(1.0, area / 10000)
        
        return normalized_area
    
    def compare_numbers(self, number1: str, number2: str) -> float:

        if not number1 or not number2:
            return 0.0
        
        if number1 == number2:
            return 1.0
        
        if number1 in number2 or number2 in number1:
            return 0.7
        
        if len(number1) == len(number2) == 1:
            return 0.3
        
        return 0.0
    
    def get_statistics(self) -> Dict:
        return {
            'enabled': self.enabled,
            'confidence_threshold': self.confidence_threshold,
            'preprocessing_enabled': self.preprocessing
        }
