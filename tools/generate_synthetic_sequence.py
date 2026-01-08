import cv2
import numpy as np
from pathlib import Path

# Config
out_dir = Path('datasets/UCSD_Anomaly_Dataset/UCSDped1/Test/TestSynthetic')
out_dir.mkdir(parents=True, exist_ok=True)
W, H = 238, 158  # Ped1 resolution
frames = 120

# Generate frames with a moving blob that speeds up mid-sequence (simulate anomaly)
for i in range(frames):
    img = np.zeros((H, W, 3), dtype=np.uint8) + 30  # dark gray background
    # position: starts slow then speeds up
    if i < frames // 2:
        x = int(20 + i * 1.2)
    else:
        x = int(20 + (frames//2)*1.2 + (i - frames//2) * 3.5)
    y = H // 2 + int(10 * np.sin(i * 0.2))
    cv2.circle(img, (max(5,min(W-5,x)), max(5,min(H-5,y))), 10, (200,200,200), -1)
    # add a few static small pedestrians-like blobs
    for j in range(5):
        cx = 30 + (j*30)
        cy = H//2 + 30 - j*10
        cv2.circle(img, (cx, cy), 4, (180,180,180), -1)
    # save
    filename = out_dir / f"frame_{i+1:04d}.tif"
    cv2.imwrite(str(filename), img)

print(f"Generated {frames} frames in {out_dir}")
