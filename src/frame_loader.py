"""
UCSD Frame loader utilities
Provides UCSDFrameLoader and helper functions used by the main detector
"""

import cv2
import os
from pathlib import Path
from typing import Tuple, Optional, List
import numpy as np

class UCSDFrameLoader:
    """Load a UCSD sequence (folder of .tif frames)"""
    def __init__(self, sequence_path: str):
        self.sequence_path = Path(sequence_path)
        if not self.sequence_path.exists():
            raise ValueError(f"Sequence path does not exist: {sequence_path}")
        
        # Gather frames (support tif, png, jpg)
        exts = ('*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg')
        frames = []
        for e in exts:
            frames.extend(sorted(self.sequence_path.glob(e)))
        frames = [f for f in frames if f.is_file()]
        if not frames:
            raise ValueError(f"No frames found in sequence: {sequence_path}")
        
        self.frames = frames
        self.total_frames = len(frames)
        self.index = -1
        # Read first frame to get shape
        first = cv2.imread(str(frames[0]))
        if first is None:
            raise ValueError(f"Unable to read first frame: {frames[0]}")
        self.height, self.width = first.shape[:2]
        # UCSD dataset recorded at ~10 FPS
        self._fps = 10.0
    
    def get_fps(self) -> float:
        return self._fps
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read next frame. Returns (ret, frame)"""
        self.index += 1
        if self.index >= self.total_frames:
            return False, None
        path = self.frames[self.index]
        frame = cv2.imread(str(path))
        if frame is None:
            return False, None
        return True, frame
    
    def reset(self):
        self.index = -1


# Helper: load ground truth if available and return loader + gt

def load_sequence_pair(sequence_path: str) -> Tuple[UCSDFrameLoader, Optional[np.ndarray]]:
    """Return (loader, ground_truth_array_or_None)"""
    seq_path = Path(sequence_path)
    loader = UCSDFrameLoader(sequence_path)
    gt = None
    # UCSD ground truth is commonly stored in a sibling folder named <SequenceName>_gt
    parent = seq_path.parent
    gt_folder = parent / f"{seq_path.name}_gt"
    # Common file: 'gt.txt' or '*.txt' with one number per line
    if gt_folder.exists() and gt_folder.is_dir():
        # try to find txt files
        txts = sorted(gt_folder.glob('*.txt'))
        if txts:
            # Read first text file and parse numbers
            try:
                arr = []
                for line in open(txts[0], 'r'):
                    line = line.strip()
                    if not line:
                        continue
                    # file may contain space separated values per line; take first
                    parts = line.split()
                    try:
                        val = int(float(parts[0]))
                    except:
                        val = 0
                    arr.append(val)
                gt = np.array(arr, dtype=np.int32)
            except Exception:
                gt = None
    return loader, gt


def get_all_sequences(dataset_path: str, split_name: str = 'Test') -> List[str]:
    """Return list of sequence folder paths under dataset_path/split_name"""
    base = Path(dataset_path) / split_name
    if not base.exists():
        return []
    seqs = [str(p) for p in sorted(base.iterdir()) if p.is_dir() and not p.name.endswith('_gt')]
    return seqs


def preview_sequence(sequence_path: str, num_frames: int = 200):
    """Quick preview: show first num_frames frames in a window (press q to quit)"""
    loader = UCSDFrameLoader(sequence_path)
    import cv2
    count = 0
    while True:
        ret, frame = loader.read_frame()
        if not ret or frame is None:
            break
        cv2.imshow('Preview', frame)
        key = cv2.waitKey(100) & 0xFF
        if key == ord('q'):
            break
        count += 1
        if count >= num_frames:
            break
    cv2.destroyAllWindows()
