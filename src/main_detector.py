"""
Main Crowd Safety Detection System
Integrates all components for UCSD dataset analysis
"""

import cv2
import numpy as np
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

from config import Config
from frame_loader import UCSDFrameLoader, load_sequence_pair
from motion_detector import MotionDetector
from density_analyzer import DensityAnalyzer
from risk_classifier import RiskClassifier

class CrowdSafetyDetector:
    """Complete crowd safety monitoring system"""
    
    def __init__(self, sequence_path: str, output_name: Optional[str] = None):
        """
        Initialize detector for a sequence
        
        Args:
            sequence_path: Path to UCSD sequence folder
            output_name: Name for output files (default: sequence name)
        """
        print(f"\n{'='*70}")
        print("AI-BASED PUBLIC SAFETY MONITORING SYSTEM")
        print("Crowd Behavior Anomaly Detection")
        print(f"{'='*70}\n")
        
        # Load sequence and ground truth
        self.loader, self.ground_truth = load_sequence_pair(sequence_path)
        self.sequence_name = Path(sequence_path).name
        self.output_name = output_name or self.sequence_name
        
        # Initialize components
        self.motion_detector = MotionDetector()
        self.density_analyzer = DensityAnalyzer()
        self.risk_classifier = RiskClassifier()
        
        # Create output directories
        Config.create_output_dirs()
        
        # Initialize logging
        self.log_file = None
        if Config.SAVE_ALERTS:
            self._init_logging()
        
        # Visualization settings
        self.visualization_mode = "combined"  # "combined", "motion", "density"
        
        print(f"✓ System initialized for: {self.sequence_name}")
        print(f"  Frames: {self.loader.total_frames}")
        print(f"  Resolution: {self.loader.width}x{self.loader.height}")
        print(f"  FPS: {self.loader.get_fps()}")
        if self.ground_truth is not None:
            print(f"  Ground truth: Available")
        print()
    
    def _init_logging(self):
        """Initialize alert logging"""
        log_path = os.path.join(Config.ALERTS_DIR, f"{self.output_name}_alerts.txt")
        # ensure alerts dir exists
        os.makedirs(Config.ALERTS_DIR, exist_ok=True)
        self.log_file = open(log_path, 'w')
        
        header = f"""
{'='*70}
AI-BASED PUBLIC SAFETY MONITORING SYSTEM
Crowd Behavior Anomaly Detection
{'='*70}
Sequence: {self.sequence_name}
Analysis Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Total Frames: {self.loader.total_frames}
FPS: {self.loader.get_fps():.2f}
{'='*70}

"""
        self.log_file.write(header)
        print(f"✓ Logging to: {log_path}")
    
    def process_sequence(self):
        """Main processing loop"""
        print("\n🎥 Processing sequence...")
        print("Press 'q' to quit, 'm' for motion view, 'd' for density view, 'c' for combined view\n")
        
        frame_num = 0
        
        # Process each frame
        while True:
            ret, frame = self.loader.read_frame()
            if not ret:
                break
            
            frame_num += 1
            
            # === DETECTION ===
            
            # 1. Motion Analysis
            motion_data = self.motion_detector.calculate_motion(frame)
            
            # 2. Density Analysis
            density_data = self.density_analyzer.calculate_density(frame)
            
            # 3. Risk Classification
            classification = self.risk_classifier.classify_risk(
                motion_data, density_data, frame_num
            )
            
            # === ALERTING ===
            if classification['is_alert']:
                self._generate_alert(classification, frame_num)
            
            # === VISUALIZATION ===
            if Config.DISPLAY_REAL_TIME or Config.SAVE_VISUALIZATIONS:
                viz_frame = self._create_visualization(
                    frame, motion_data, density_data, classification
                )
                
                if Config.DISPLAY_REAL_TIME:
                    cv2.imshow('Crowd Safety Monitoring', viz_frame)
                
                if Config.SAVE_VISUALIZATIONS and classification['is_alert']:
                    self._save_frame(viz_frame, frame_num)
            
            # === USER INTERACTION ===
            if Config.DISPLAY_REAL_TIME:
                key = cv2.waitKey(30) & 0xFF
                
                if key == ord('q'):
                    print("\n⚠ User interrupted")
                    break
                elif key == ord('m'):
                    self.visualization_mode = "motion"
                elif key == ord('d'):
                    self.visualization_mode = "density"
                elif key == ord('c'):
                    self.visualization_mode = "combined"
            
            # Progress indicator
            if frame_num % 50 == 0:
                progress = (frame_num / self.loader.total_frames) * 100
                print(f"  Progress: {frame_num}/{self.loader.total_frames} ({progress:.1f}%)")
        
        # === FINALIZATION ===
        self._generate_summary()
        self._cleanup()
    
    def _generate_alert(self, classification: Dict, frame_num: int):
        """Generate and log alert"""
        alert_msg = f"""
{'='*70}
ALERT - {classification['risk_level']} RISK
{'='*70}
Timestamp: {classification['timestamp']}
Frame: {frame_num}/{self.loader.total_frames}
Video Time: {frame_num/self.loader.get_fps():.2f}s
Confidence: {classification['confidence']*100:.1f}%
{'='*70}
PRIMARY CAUSE:
  {classification['primary_cause']}

METRICS:
  Motion Level: {classification['metrics']['motion']:.2f}
  Crowd Density: {classification['metrics']['density']*100:.1f}%
  Motion Variance: {classification['metrics']['motion_std']:.2f}

CONTRIBUTING FACTORS:
"""
        
        for factor in classification['contributing_factors']:
            alert_msg += f"  • {factor}\n"
        
        alert_msg += f"{'='*70}\n"
        
        # Console output
        print(alert_msg)
        
        # File logging
        if self.log_file:
            self.log_file.write(alert_msg + "\n")
            self.log_file.flush()
    
    def _create_visualization(self, frame: np.ndarray, motion_data: Dict, 
                             density_data: Dict, classification: Dict) -> np.ndarray:
        """Create visualization frame"""
        
        # Risk level colors
        colors = {
            "CRITICAL": (255, 0, 255),  # Purple
            "HIGH": (0, 0, 255),        # Red
            "MEDIUM": (0, 165, 255),    # Orange
            "LOW": (0, 255, 255),       # Yellow
            "NORMAL": (0, 255, 0)       # Green
        }
        
        color = colors.get(classification['risk_level'], (255, 255, 255))
        
        # Create different visualizations based on mode
        if self.visualization_mode == "motion":
            viz = self.motion_detector.visualize_flow(
                frame, motion_data.get('flow'), motion_data.get('magnitude')
            )
        elif self.visualization_mode == "density":
            viz = self.density_analyzer.visualize_density(
                frame, 
                density_data.get('fg_mask'),
                density_data.get('overall_density'),
                density_data.get('density_center')
            )
        else:  # combined
            viz = frame.copy()
        
        # Alert border
        if classification['is_alert']:
            thickness = 15 if classification['risk_level'] == "CRITICAL" else 10
            cv2.rectangle(viz, (0, 0), 
                         (viz.shape[1]-1, viz.shape[0]-1), 
                         color, thickness)
        
        # Info overlay
        y_offset = 30
        line_height = 35
        
        # Risk level (large)
        cv2.putText(viz, f"RISK: {classification['risk_level']}", 
                   (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        y_offset += line_height
        
        # Metrics
        cv2.putText(viz, f"Motion: {motion_data['avg_motion']:.1f}", 
                   (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        y_offset += line_height
        
        cv2.putText(viz, f"Density: {density_data['overall_density']*100:.1f}%", 
                   (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        y_offset += line_height
        
        cv2.putText(viz, f"Frame: {classification['frame_num']}/{self.loader.total_frames}", 
                   (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
        
        # Mode indicator
        cv2.putText(viz, f"View: {self.visualization_mode.upper()}", 
                   (15, viz.shape[0] - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)
        
        return viz
    
    def _save_frame(self, frame: np.ndarray, frame_num: int):
        """Save visualization frame"""
        output_path = os.path.join(
            Config.VIZ_DIR, 
            f"{self.output_name}_frame_{frame_num:04d}.jpg"
        )
        # ensure dir
        os.makedirs(Config.VIZ_DIR, exist_ok=True)
        cv2.imwrite(output_path, frame)
    
    def _generate_summary(self):
        """Generate final summary and metrics"""
        print(f"\n{'='*70}")
        print("GENERATING SUMMARY...")
        print(f"{'='*70}\n")
        
        # Get alert summary
        alert_summary = self.risk_classifier.get_alert_summary()
        
        summary = f"""
{'='*70}
DETECTION SUMMARY
{'='*70}
Sequence: {self.sequence_name}
Total Frames Processed: {alert_summary['total_frames']}
Total Alerts Generated: {alert_summary['total_alerts']}
Video Duration: {alert_summary['total_frames']/self.loader.get_fps():.2f} seconds
Alert Rate: {alert_summary['alert_rate']:.3f} alerts/frame

RISK DISTRIBUTION:
"""
        
        for risk_level, percentage in alert_summary['risk_percentages'].items():
            count = alert_summary['risk_distribution'][risk_level]
            summary += f"  {risk_level:10s}: {count:4d} frames ({percentage:5.1f}%)\n"
        
        # Ground truth comparison
        if self.ground_truth is not None and Config.CALCULATE_METRICS:
            metrics = self.risk_classifier.compare_with_ground_truth(self.ground_truth)
            
            if metrics:
                summary += f"""
{'='*70}
PERFORMANCE METRICS (vs Ground Truth)
{'='*70}
Accuracy:  {metrics['accuracy']*100:.2f}%
Precision: {metrics['precision']*100:.2f}%
Recall:    {metrics['recall']*100:.2f}%
F1-Score:  {metrics['f1_score']*100:.2f}%

Confusion Matrix:
  True Positives:  {metrics['true_positives']}
  False Positives: {metrics['false_positives']}
  True Negatives:  {metrics['true_negatives']}
  False Negatives: {metrics['false_negatives']}
"""
        
        summary += f"""
{'='*70}
Analysis Completed: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
{'='*70}
"""
        
        print(summary)
        
        if self.log_file:
            self.log_file.write(summary)
        
        # Save summary to separate file
        summary_path = os.path.join(Config.RESULTS_DIR, f"{self.output_name}_summary.txt")
        os.makedirs(Config.RESULTS_DIR, exist_ok=True)
        with open(summary_path, 'w') as f:
            f.write(summary)
        
        print(f"\n✅ Analysis complete!")
        print(f"📄 Results saved to:")
        print(f"   - {os.path.join(Config.ALERTS_DIR, f'{self.output_name}_alerts.txt')}")
        print(f"   - {summary_path}")
    
    def _cleanup(self):
        """Clean up resources"""
        if self.log_file:
            self.log_file.close()
        
        cv2.destroyAllWindows()


# ==================== BATCH PROCESSING ====================

def process_multiple_sequences(sequence_paths: list, output_dir: Optional[str] = None):
    """
    Process multiple sequences in batch
    
    Args:
        sequence_paths: List of sequence folder paths
        output_dir: Optional output directory
    """
    print(f"\n{'='*70}")
    print(f"BATCH PROCESSING: {len(sequence_paths)} sequences")
    print(f"{'='*70}\n")
    
    results = []
    
    for i, seq_path in enumerate(sequence_paths, 1):
        print(f"\n[{i}/{len(sequence_paths)}] Processing: {Path(seq_path).name}")
        print("-" * 70)
        
        try:
            detector = CrowdSafetyDetector(seq_path)
            detector.process_sequence()
            
            summary = detector.risk_classifier.get_alert_summary()
            results.append({
                'sequence': Path(seq_path).name,
                'status': 'SUCCESS',
                'alerts': summary['total_alerts'],
                'frames': summary['total_frames']
            })
            
        except Exception as e:
            print(f"❌ Error processing {seq_path}: {e}")
            results.append({
                'sequence': Path(seq_path).name,
                'status': 'FAILED',
                'error': str(e)
            })
    
    # Final summary
    print(f"\n{'='*70}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*70}\n")
    
    for result in results:
        status_symbol = "✓" if result['status'] == 'SUCCESS' else "✗"
        print(f"{status_symbol} {result['sequence']}: {result['status']}")
        if result['status'] == 'SUCCESS':
            print(f"   Alerts: {result['alerts']}, Frames: {result['frames']}")
