import os

class Config:
    # ==================== DATASET PATHS ====================
    DATASET_ROOT = "datasets/UCSD_Anomaly_Dataset"
    
    # Peds1 paths
    PEDS1_TRAIN = os.path.join(DATASET_ROOT, "UCSDped1/Train")
    PEDS1_TEST = os.path.join(DATASET_ROOT, "UCSDped1/Test")
    PEDS1_GT = os.path.join(DATASET_ROOT, "UCSDped1/Test")
    
    # Peds2 paths
    PEDS2_TRAIN = os.path.join(DATASET_ROOT, "UCSDped2/Train")
    PEDS2_TEST = os.path.join(DATASET_ROOT, "UCSDped2/Test")
    PEDS2_GT = os.path.join(DATASET_ROOT, "UCSDped2/Test")
    
    # ==================== DETECTION THRESHOLDS ====================
    # Motion thresholds (optimized for UCSD dataset)
    MOTION_LOW = 8.0        # Normal walking
    MOTION_MEDIUM = 15.0    # Faster movement (bikes/skaters start here)
    MOTION_HIGH = 25.0      # Rapid movement (bikes/skaters/carts)
    
    # Density thresholds
    DENSITY_LOW = 0.15      # Sparse crowd
    DENSITY_MEDIUM = 0.35   # Moderate crowd
    DENSITY_HIGH = 0.55     # Dense crowd
    
    # Temporal smoothing (reduce false positives)
    TEMPORAL_WINDOW = 5     # Frames to average over
    ALERT_THRESHOLD = 3     # Consecutive anomalous frames needed
    
    # ==================== OPTICAL FLOW PARAMETERS ====================
    FLOW_PYR_SCALE = 0.5
    FLOW_LEVELS = 3
    FLOW_WINSIZE = 15
    FLOW_ITERATIONS = 3
    FLOW_POLY_N = 5
    FLOW_POLY_SIGMA = 1.2
    
    # ==================== BACKGROUND SUBTRACTION ====================
    BG_HISTORY = 500
    BG_VAR_THRESHOLD = 16
    BG_DETECT_SHADOWS = False
    
    # ==================== OUTPUT SETTINGS ====================
    OUTPUT_DIR = "outputs"
    ALERTS_DIR = os.path.join(OUTPUT_DIR, "alerts")
    VIZ_DIR = os.path.join(OUTPUT_DIR, "visualizations")
    RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
    
    SAVE_VISUALIZATIONS = True
    SAVE_ALERTS = True
    DISPLAY_REAL_TIME = True
    
    # ==================== EVALUATION SETTINGS ====================
    USE_GROUND_TRUTH = True  # Compare against UCSD ground truth
    CALCULATE_METRICS = True # Calculate precision, recall, F1
    
    # ==================== PROCESSING SETTINGS ====================
    BATCH_SIZE = 1           # Process one sequence at a time
    SKIP_FRAMES = 1          # Process every frame (no skipping)
    MAX_FRAMES = None        # Process all frames (set number to limit)
    
    @staticmethod
    def create_output_dirs():
        """Create output directories if they don't exist"""
        os.makedirs(Config.ALERTS_DIR, exist_ok=True)
        os.makedirs(Config.VIZ_DIR, exist_ok=True)
        os.makedirs(Config.RESULTS_DIR, exist_ok=True)
