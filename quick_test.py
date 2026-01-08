"""
Quick test script to verify your setup and see system in action
Run this first to ensure everything works
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, 'src')

def test_imports():
    """Test if all required libraries are installed"""
    print("\n" + "="*70)
    print("STEP 1: Testing Imports...")
    print("="*70)
    
    try:
        import cv2
        print(f"✓ OpenCV version: {cv2.__version__}")
    except ImportError:
        print("✗ OpenCV not installed. Run: pip install opencv-python")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy version: {np.__version__}")
    except ImportError:
        print("✗ NumPy not installed. Run: pip install numpy")
        return False
    
    print("✓ All imports successful!\n")
    return True

def test_dataset():
    """Check if UCSD dataset is present"""
    print("="*70)
    print("STEP 2: Checking Dataset...")
    print("="*70)
    
    dataset_path = Path("datasets/UCSD_Anomaly_Dataset")
    
    if not dataset_path.exists():
        print(f"✗ Dataset not found at: {dataset_path}")
        print("\nPlease download UCSD dataset from:")
        print("http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm")
        print("\nExtract to: datasets/UCSD_Anomaly_Dataset/")
        return False
    
    # Check Ped1
    ped1_test = dataset_path / "UCSDped1" / "Test"
    if ped1_test.exists():
        sequences = list(ped1_test.glob("Test*"))
        sequences = [s for s in sequences if s.is_dir() and not s.name.endswith("_gt")]
        print(f"✓ Found UCSDped1: {len(sequences)} test sequences")
    else:
        print("✗ UCSDped1 not found")
        return False
    
    # Check Ped2
    ped2_test = dataset_path / "UCSDped2" / "Test"
    if ped2_test.exists():
        sequences = list(ped2_test.glob("Test*"))
        sequences = [s for s in sequences if s.is_dir() and not s.name.endswith("_gt")]
        print(f"✓ Found UCSDped2: {len(sequences)} test sequences")
    else:
        print("⚠ UCSDped2 not found (optional)")
    
    print("✓ Dataset check complete!\n")
    return True

def test_frame_loading():
    """Test loading a single sequence"""
    print("="*70)
    print("STEP 3: Testing Frame Loader...")
    print("="*70)
    
    try:
        from frame_loader import UCSDFrameLoader
        
        # Find first test sequence
        test_seq = Path("datasets/UCSD_Anomaly_Dataset/UCSDped1/Test/Test001")
        
        if not test_seq.exists():
            print(f"✗ Test sequence not found: {test_seq}")
            return False
        
        loader = UCSDFrameLoader(str(test_seq))
        
        # Try reading first frame
        ret, frame = loader.read_frame()
        
        if ret and frame is not None:
            print(f"✓ Successfully loaded frame")
            print(f"  Shape: {frame.shape}")
            print(f"  Total frames: {loader.total_frames}")
        else:
            print("✗ Failed to read frame")
            return False
        
        loader.reset()
        print("✓ Frame loader working!\n")
        return True
        
    except Exception as e:
        print(f"✗ Error testing frame loader: {e}")
        return False

def run_quick_demo():
    """Run a quick demo on first 50 frames"""
    print("="*70)
    print("STEP 4: Running Quick Demo (50 frames)...")
    print("="*70)
    print("Press 'q' to skip demo\n")
    
    try:
        from main_detector import CrowdSafetyDetector
        from config import Config
        
        # Reduce output for quick test
        Config.SAVE_VISUALIZATIONS = False
        Config.MAX_FRAMES = 50
        
        test_seq = "datasets/UCSD_Anomaly_Dataset/UCSDped1/Test/Test001"
        
        detector = CrowdSafetyDetector(test_seq, output_name="quick_test")
        detector.process_sequence()
        
        print("\n✓ Demo complete!")
        print("Check outputs/alerts/quick_test_alerts.txt for results")
        return True
        
    except Exception as e:
        print(f"✗ Error running demo: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║     CROWD SAFETY SYSTEM - QUICK TEST                              ║
║     Verify your setup before running full system                  ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
""")
    
    # Run tests
    tests = [
        ("Imports", test_imports),
        ("Dataset", test_dataset),
        ("Frame Loading", test_frame_loading),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            
            if not result:
                print(f"\n⚠ {test_name} test failed. Please fix before proceeding.\n")
                return
        except Exception as e:
            print(f"\n✗ {test_name} test crashed: {e}\n")
            results.append((test_name, False))
            return
    
    # All tests passed, offer demo
    print("\n" + "="*70)
    print("ALL TESTS PASSED! ✓")
    print("="*70)
    
    response = input("\nRun quick demo? (y/n): ").strip().lower()
    
    if response == 'y':
        run_quick_demo()
    else:
        print("\nSkipping demo. You're ready to run the full system!")
        print("\nNext steps:")
        print("1. Process single sequence:")
        print("   python run_detection.py --sequence datasets/UCSD_Anomaly_Dataset/UCSDped1/Test/Test001")
        print("\n2. Process all sequences:")
        print("   python run_detection.py --dataset ped1 --split test --batch")

if __name__ == "__main__":
    main()
