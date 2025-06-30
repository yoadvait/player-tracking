import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import mediapipe as mp
from pathlib import Path

class PartBasedFeatureExtractor(nn.Module):
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config['advanced_reid']
        self.feature_dim = self.config['feature_dim']
        self.parts = self.config['parts']
        
        self.backbone = self._create_backbone()
        
        self.part_heads = nn.ModuleDict({
            part: nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(self.backbone_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256)
            ) for part in self.parts
        })
        
        self.global_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.backbone_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        total_part_dim = len(self.parts) * 256 + 256  
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_part_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, self.feature_dim)
        )
        
    def _create_backbone(self):
        """Create backbone network"""
        backbone_name = self.config['deep_features']['backbone']
        
        if backbone_name == 'resnet50':
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.backbone_dim = 2048
            backbone = nn.Sequential(*list(backbone.children())[:-2])
        elif backbone_name == 'mobilenet':
            backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
            self.backbone_dim = 960
            backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        frozen_layers = self.config['deep_features'].get('frozen_layers', 0)
        if frozen_layers > 0:
            for i, child in enumerate(backbone.children()):
                if i < frozen_layers:
                    for param in child.parameters():
                        param.requires_grad = False
        
        return backbone
    
    def forward(self, x, part_masks=None):

        backbone_features = self.backbone(x)  
        
        global_features = self.global_head(backbone_features)
        
        part_features = []
        
        if part_masks is not None:
            for i, part in enumerate(self.parts):
                part_mask = part_masks[:, i:i+1, :, :]
                
                part_mask_resized = nn.functional.interpolate(
                    part_mask, size=backbone_features.shape[2:], mode='bilinear'
                )
                
                masked_features = backbone_features * part_mask_resized
                part_feat = self.part_heads[part](masked_features)
                part_features.append(part_feat)
        else:
            h, w = backbone_features.shape[2], backbone_features.shape[3]
            
            part_regions = self._get_default_part_regions(h, w)
            
            for part, (y1, y2, x1, x2) in zip(self.parts, part_regions):
                region_features = backbone_features[:, :, y1:y2, x1:x2]
                part_feat = self.part_heads[part](region_features)
                part_features.append(part_feat)
        
        all_features = torch.cat([global_features] + part_features, dim=1)
        
        fused_features = self.fusion_layer(all_features)
        
        final_features = F.normalize(fused_features, p=2, dim=1)
        
        return final_features, {
            'global': global_features,
            'parts': {part: feat for part, feat in zip(self.parts, part_features)},
            'backbone': backbone_features
        }
    
    def _get_default_part_regions(self, h, w):
        regions = []
        
        part_height = h // len(self.parts)
        
        for i in range(len(self.parts)):
            y1 = i * part_height
            y2 = (i + 1) * part_height if i < len(self.parts) - 1 else h
            x1, x2 = 0, w
            regions.append((y1, y2, x1, x2))
        
        return regions

class DeepFeatureExtractor:
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        feature_extractor_type = config.get('advanced_reid', {}).get('feature_extractor', 'resnet50')
        self._use_lightweight = (feature_extractor_type == 'lightweight')
        
        if self._use_lightweight:
            self.logger.info("🚀 Initializing in lightweight feature extraction mode")
            self.device = None
            self.feature_model = None
            self.pose_estimator = self._load_pose_estimator()
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.feature_model = self._load_feature_model()
            self.pose_estimator = self._load_pose_estimator()
            
            self.preprocess = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((384, 128)),  
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        
        self.logger.info(f"✅ Deep feature extractor initialized ({'lightweight' if self._use_lightweight else 'on ' + str(self.device)})")
    
    def _load_feature_model(self):
        model = PartBasedFeatureExtractor(self.config)
        model.to(self.device)
        model.eval()
        
        weights_path = Path("models/reid_weights.pth")
        if weights_path.exists():
            try:
                model.load_state_dict(torch.load(weights_path, map_location=self.device))
                self.logger.info("✅ Loaded pre-trained ReID weights")
            except Exception as e:
                self.logger.warning(f"⚠️  Could not load pre-trained weights: {e}")
        
        return model
    
    def _load_pose_estimator(self):
        if self._use_lightweight:
            return None
            
        if not self.config.get('pose_estimation', {}).get('enabled', False):
            return None
        
        try:
            mp_pose = mp.solutions.pose
            pose_estimator = mp_pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.3
            )
            self.logger.info("✅ Pose estimator loaded")
            return pose_estimator
        except Exception as e:
            self.logger.warning(f"⚠️  Could not load pose estimator: {e}")
            return None
    
    def extract_features(self, frame: np.ndarray, bbox: np.ndarray, track_id: int = None) -> Dict:

        try:
            x1, y1, x2, y2 = bbox.astype(int)
            player_crop = frame[y1:y2, x1:x2]
            
            if player_crop.size == 0:
                return self._empty_features()
            
            features = {}
            
            deep_features = self._extract_deep_features(player_crop)
            features['deep'] = deep_features
            
            color_features = self._extract_color_features(player_crop)
            features['color'] = color_features
            
            if self.pose_estimator:
                pose_features = self._extract_pose_features(player_crop)
                features['pose'] = pose_features
            
            texture_features = self._extract_texture_features(player_crop)
            features['texture'] = texture_features
            
            combined_features = self._combine_features(features)
            features['combined'] = combined_features
            
            return features
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return self._empty_features()
    
    def _extract_deep_features(self, crop: np.ndarray) -> np.ndarray:
        try:
            if hasattr(self, '_use_lightweight') and self._use_lightweight:
                return self._extract_lightweight_features(crop)
            
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            input_tensor = self.preprocess(crop_rgb).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features, _ = self.feature_model(input_tensor)
                features = features.cpu().numpy().flatten()
            
            return features
            
        except Exception as e:
            self.logger.warning(f"Deep feature extraction failed: {e}")
            return self._extract_lightweight_features(crop)
    
    def _extract_lightweight_features(self, crop: np.ndarray) -> np.ndarray:
        """Extract lightweight features as fallback when deep learning is too slow"""
        try:
            crop_resized = cv2.resize(crop, (64, 128))
            
            features = []
            
            hsv = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2HSV)
            for i in range(3):
                hist = cv2.calcHist([hsv], [i], None, [32], [0, 256])
                features.extend(hist.flatten())
            
            gray = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)
            
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            features.extend([
                np.mean(grad_x), np.std(grad_x), np.mean(np.abs(grad_x)),
                np.mean(grad_y), np.std(grad_y), np.mean(np.abs(grad_y))
            ])
            
            h, w = gray.shape
            regions = [
                gray[:h//2, :w//2],      
                gray[:h//2, w//2:],      
                gray[h//2:, :w//2],      
                gray[h//2:, w//2:]       
            ]
            
            for region in regions:
                if region.size > 0:
                    features.extend([np.mean(region), np.std(region)])
                else:
                    features.extend([0.0, 0.0])
            
            features = np.array(features, dtype=np.float32)
            
            expected_dim = self.config['advanced_reid']['feature_dim']
            if len(features) < expected_dim:
                features = np.pad(features, (0, expected_dim - len(features)), 'constant')
            elif len(features) > expected_dim:
                features = features[:expected_dim]
            
            features = features / (np.linalg.norm(features) + 1e-6)
            
            return features
            
        except Exception as e:
            self.logger.warning(f"Lightweight feature extraction failed: {e}")
            return np.random.randn(self.config['advanced_reid']['feature_dim']) * 0.01
    
    def enable_lightweight_mode(self):
        self._use_lightweight = True
        self.logger.info("🚀 Enabled lightweight feature extraction mode")
    
    def _extract_color_features(self, crop: np.ndarray) -> np.ndarray:
        try:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            
            h_hist = cv2.calcHist([hsv], [0], None, [50], [0, 180])
            s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256])
            v_hist = cv2.calcHist([hsv], [2], None, [32], [0, 256])
            
            h_hist = h_hist.flatten() / (h_hist.sum() + 1e-7)
            s_hist = s_hist.flatten() / (s_hist.sum() + 1e-7)
            v_hist = v_hist.flatten() / (v_hist.sum() + 1e-7)
            
            color_features = np.concatenate([h_hist, s_hist, v_hist])
            
            return color_features
            
        except Exception as e:
            self.logger.warning(f"Color feature extraction failed: {e}")
            return np.zeros(114)
    
    def _extract_pose_features(self, crop: np.ndarray) -> Optional[np.ndarray]:
        if not self.pose_estimator:
            return None
        
        try:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            results = self.pose_estimator.process(crop_rgb)
            
            if results.pose_landmarks:
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.visibility])
                
                return np.array(landmarks)
            else:
                return np.zeros(33 * 3)
                
        except Exception as e:
            self.logger.warning(f"Pose feature extraction failed: {e}")
            return np.zeros(33 * 3)
    
    def _extract_texture_features(self, crop: np.ndarray) -> np.ndarray:
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            
            gray = cv2.resize(gray, (64, 128))
            
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            texture_features = [
                np.mean(grad_x), np.std(grad_x),
                np.mean(grad_y), np.std(grad_y),
                np.mean(np.abs(grad_x)), np.mean(np.abs(grad_y))
            ]
            
            return np.array(texture_features)
            
        except Exception as e:
            self.logger.warning(f"Texture feature extraction failed: {e}")
            return np.zeros(6)
    
    def _combine_features(self, features: Dict) -> np.ndarray:
        combined = []
        
        if 'deep' in features:
            combined.append(features['deep'])
        
        if 'color' in features:
            combined.append(features['color'])
        
        if 'pose' in features and features['pose'] is not None:
            combined.append(features['pose'])
        
        if 'texture' in features:
            combined.append(features['texture'])
        
        if combined:
            return np.concatenate(combined)
        else:
            return np.random.randn(self.config['advanced_reid']['feature_dim']) * 0.01
    
    def _empty_features(self) -> Dict:
        return {
            'deep': np.random.randn(self.config['advanced_reid']['feature_dim']) * 0.01,
            'color': np.zeros(114),
            'pose': np.zeros(33 * 3),
            'texture': np.zeros(6),
            'combined': np.random.randn(self.config['advanced_reid']['feature_dim'] + 114 + 99 + 6) * 0.01
        }