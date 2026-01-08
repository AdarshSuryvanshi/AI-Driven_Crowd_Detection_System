"""
Risk classification engine
Combines motion and density metrics to classify safety risks
"""

import numpy as np
from typing import Dict
from config import Config
from datetime import datetime

class RiskClassifier:
    """Classify crowd safety risks based on motion and density"""
    
    def __init__(self):
        self.alert_history = []
        self.consecutive_alerts = 0
        self.frame_classifications = []
    
    def classify_risk(self, motion_data: Dict, density_data: Dict, frame_num: int) -> Dict:
        """
        Classify risk level based on motion and density
        
        Args:
            motion_data: Dictionary from MotionDetector
            density_data: Dictionary from DensityAnalyzer
            frame_num: Current frame number
        
        Returns:
            Classification result dictionary with:
            - risk_level: "NORMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"
            - confidence: 0-1 confidence score
            - primary_cause: Main reason for classification
            - contributing_factors: List of contributing factors
            - is_alert: Boolean whether to trigger alert
        """
        # Extract metrics
        motion = motion_data.get('smoothed_motion', 0)
        motion_std = motion_data.get('motion_std', 0)
        density = density_data.get('smoothed_density', 0)
        
        # Initialize result
        result = {
            'frame_num': frame_num,
            'timestamp': datetime.now().strftime("%H:%M:%S.%f")[:-3],
            'risk_level': "NORMAL",
            'confidence': 0.0,
            'primary_cause': "Normal crowd behavior",
            'contributing_factors': [],
            'is_alert': False,
            'metrics': {
                'motion': motion,
                'density': density,
                'motion_std': motion_std
            }
        }
        
        # Classification logic
        factors = []
        score = 0.0
        
        # === CRITICAL RISK ===
        # Very fast movement + high density = stampede/panic
        if motion >= Config.MOTION_HIGH and density >= Config.DENSITY_HIGH:
            result['risk_level'] = "CRITICAL"
            result['primary_cause'] = "STAMPEDE RISK: Rapid movement in overcrowded area"
            result['confidence'] = 0.95
            score = 10.0
            factors.append(f"Extreme motion detected (motion={motion:.1f})")
            factors.append(f"Overcrowding present (density={density*100:.1f}%)")
        
        # === HIGH RISK ===
        # Very fast movement (bikes, skaters, running)
        elif motion >= Config.MOTION_HIGH:
            result['risk_level'] = "HIGH"
            result['primary_cause'] = "Non-pedestrian entity or panic behavior detected"
            result['confidence'] = 0.85
            score = 8.0
            factors.append(f"High-speed movement (motion={motion:.1f})")
            
            if density >= Config.DENSITY_MEDIUM:
                factors.append(f"Elevated crowd density (density={density*100:.1f}%)")
        
        # Dangerous crowding + moderate movement
        elif density >= Config.DENSITY_HIGH and motion >= Config.MOTION_MEDIUM:
            result['risk_level'] = "HIGH"
            result['primary_cause'] = "Dangerous movement in overcrowded area"
            result['confidence'] = 0.80
            score = 7.5
            factors.append(f"Severe overcrowding (density={density*100:.1f}%)")
            factors.append(f"Unusual movement pattern (motion={motion:.1f})")
        
        # === MEDIUM RISK ===
        # Moderate motion + moderate density
        elif motion >= Config.MOTION_MEDIUM and density >= Config.DENSITY_MEDIUM:
            result['risk_level'] = "MEDIUM"
            result['primary_cause'] = "Unusual activity in moderately crowded area"
            result['confidence'] = 0.70
            score = 5.0
            factors.append(f"Irregular movement (motion={motion:.1f})")
            factors.append(f"Moderate crowding (density={density*100:.1f}%)")
        
        # Fast movement alone
        elif motion >= Config.MOTION_MEDIUM:
            result['risk_level'] = "MEDIUM"
            result['primary_cause'] = "Irregular crowd movement detected"
            result['confidence'] = 0.65
            score = 4.5
            factors.append(f"Faster than normal movement (motion={motion:.1f})")
        
        # High density alone
        elif density >= Config.DENSITY_HIGH:
            result['risk_level'] = "MEDIUM"
            result['primary_cause'] = "High crowd density detected"
            result['confidence'] = 0.60
            score = 4.0
            factors.append(f"Overcrowding concern (density={density*100:.1f}%)")
        
        # === LOW RISK ===
        # Slight irregularity
        elif motion >= Config.MOTION_LOW or density >= Config.DENSITY_MEDIUM:
            result['risk_level'] = "LOW"
            result['primary_cause'] = "Minor irregular activity"
            result['confidence'] = 0.50
            score = 2.0
            
            if motion >= Config.MOTION_LOW:
                factors.append(f"Slightly elevated movement (motion={motion:.1f})")
            if density >= Config.DENSITY_MEDIUM:
                factors.append(f"Moderate density (density={density*100:.1f}%)")
        
        # === NORMAL ===
        else:
            result['risk_level'] = "NORMAL"
            result['confidence'] = 0.90
            score = 0.0
        
        # Add contributing factors
        result['contributing_factors'] = factors
        
        # Alert decision with temporal filtering
        if result['risk_level'] in ["MEDIUM", "HIGH", "CRITICAL"]:
            self.consecutive_alerts += 1
        else:
            self.consecutive_alerts = 0
        
        # Only trigger alert after consecutive detections
        result['is_alert'] = (self.consecutive_alerts >= Config.ALERT_THRESHOLD)
        
        # Store classification
        self.frame_classifications.append(result)
        
        return result
    
    def get_alert_summary(self) -> Dict:
        """
        Generate summary of all alerts
        
        Returns:
            Summary statistics
        """
        if not self.frame_classifications:
            return {
                'total_frames': 0,
                'total_alerts': 0,
                'risk_distribution': {}
            }
        
        total_frames = len(self.frame_classifications)
        
        # Count by risk level
        risk_counts = {
            'NORMAL': 0,
            'LOW': 0,
            'MEDIUM': 0,
            'HIGH': 0,
            'CRITICAL': 0
        }
        
        total_alerts = 0
        
        for classification in self.frame_classifications:
            risk_level = classification['risk_level']
            risk_counts[risk_level] += 1
            
            if classification['is_alert']:
                total_alerts += 1
        
        return {
            'total_frames': total_frames,
            'total_alerts': total_alerts,
            'alert_rate': total_alerts / total_frames if total_frames > 0 else 0,
            'risk_distribution': risk_counts,
            'risk_percentages': {
                k: (v / total_frames * 100) if total_frames > 0 else 0 
                for k, v in risk_counts.items()
            }
        }
    
    def compare_with_ground_truth(self, ground_truth: np.ndarray) -> Dict:
        """
        Compare classifications with ground truth labels
        
        Args:
            ground_truth: Binary array (1=anomaly, 0=normal)
        
        Returns:
            Performance metrics (precision, recall, F1)
        """
        if ground_truth is None or len(self.frame_classifications) == 0:
            return None
        
        # Convert classifications to binary predictions
        predictions = []
        for classification in self.frame_classifications:
            # Consider MEDIUM/HIGH/CRITICAL as anomaly
            is_anomaly = classification['risk_level'] in ["MEDIUM", "HIGH", "CRITICAL"]
            predictions.append(1 if is_anomaly else 0)
        
        predictions = np.array(predictions)
        
        # Ensure same length
        min_len = min(len(predictions), len(ground_truth))
        predictions = predictions[:min_len]
        gt = ground_truth[:min_len]
        
        # Calculate metrics
        tp = np.sum((predictions == 1) & (gt == 1))
        fp = np.sum((predictions == 1) & (gt == 0))
        fn = np.sum((predictions == 0) & (gt == 1))
        tn = np.sum((predictions == 0) & (gt == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn)
        }
    
    def reset(self):
        """Reset classifier state"""
        self.alert_history = []
        self.consecutive_alerts = 0
        self.frame_classifications = []
