"""
Optical flow based motion detector
"""

import cv2
import numpy as np
from typing import Dict, Optional
from config import Config

class MotionDetector:
    def __init__(self):
        self.prev_gray = None
        self.motion_history = []
        self.window = Config.TEMPORAL_WINDOW
    
    def calculate_motion(self, frame: np.ndarray) -> Dict:
        """Calculate optical flow-based motion metrics for the given BGR frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            # initialize outputs with zeros
            out = {
                'flow': None,
                'magnitude': np.zeros_like(gray, dtype=np.float32),
                'avg_motion': 0.0,
                'motion_std': 0.0,
                'smoothed_motion': 0.0
            }
            return out
        # calc optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            Config.FLOW_PYR_SCALE, Config.FLOW_LEVELS,
            Config.FLOW_WINSIZE, Config.FLOW_ITERATIONS,
            Config.FLOW_POLY_N, Config.FLOW_POLY_SIGMA, 0
        )
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        avg_motion = float(np.mean(mag))
        motion_std = float(np.std(mag))
        # update history and smoothing
        self.motion_history.append(avg_motion)
        if len(self.motion_history) > self.window:
            self.motion_history.pop(0)
        smoothed = float(np.mean(self.motion_history)) if self.motion_history else avg_motion
        # store prev
        self.prev_gray = gray
        return {
            'flow': flow,
            'magnitude': mag,
            'avg_motion': avg_motion,
            'motion_std': motion_std,
            'smoothed_motion': smoothed
        }
    
    def visualize_flow(self, frame, flow, magnitude=None, step=8):
        """Render flow as colored lines over the image"""
        if flow is None:
            return frame
        h, w = frame.shape[:2]
        vis = frame.copy()
        # draw sparse sampling
        y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
        fx, fy = flow[y, x].T
        # draw lines
        for (x1, y1, dx, dy) in zip(x, y, fx, fy):
            x2 = int(x1 + dx)
            y2 = int(y1 + dy)
            cv2.line(vis, (x1, y1), (x2, y2), (0,255,0), 1)
            cv2.circle(vis, (x1, y1), 1, (0,255,0), -1)
        return vis
