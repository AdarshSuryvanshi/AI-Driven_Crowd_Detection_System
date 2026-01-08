"""
Background-subtraction based density analyzer
"""

import cv2
import numpy as np
from typing import Dict
from config import Config

class DensityAnalyzer:
    def __init__(self):
        # create background subtractor
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=Config.BG_HISTORY, varThreshold=Config.BG_VAR_THRESHOLD,
            detectShadows=Config.BG_DETECT_SHADOWS
        )
        self.density_history = []
        self.window = Config.TEMPORAL_WINDOW
    
    def calculate_density(self, frame) -> Dict:
        """Return foreground mask and density metrics"""
        fg = self.bg.apply(frame)
        # threshold to binary
        _, fg_bin = cv2.threshold(fg, 127, 255, cv2.THRESH_BINARY)
        total_pixels = frame.shape[0] * frame.shape[1]
        fg_pixels = int(np.count_nonzero(fg_bin))
        overall_density = fg_pixels / float(total_pixels)
        # smoothing
        self.density_history.append(overall_density)
        if len(self.density_history) > self.window:
            self.density_history.pop(0)
        smoothed = float(np.mean(self.density_history)) if self.density_history else overall_density
        # compute density center (centroid) if fg present
        density_center = None
        if fg_pixels > 0:
            moments = cv2.moments(fg_bin)
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                density_center = (cx, cy)
        return {
            'fg_mask': fg_bin,
            'overall_density': overall_density,
            'smoothed_density': smoothed,
            'density_center': density_center
        }
    
    def visualize_density(self, frame, fg_mask, overall_density, density_center):
        if fg_mask is None:
            return frame
        vis = frame.copy()
        # overlay mask as red transparent
        mask_colored = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
        alpha = 0.5
        vis = cv2.addWeighted(vis, 1.0, mask_colored, alpha, 0)
        # draw center
        if density_center is not None:
            cv2.circle(vis, density_center, 6, (0,0,255), -1)
        # put density text
        cv2.putText(vis, f"Density: {overall_density*100:.1f}%", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        return vis
