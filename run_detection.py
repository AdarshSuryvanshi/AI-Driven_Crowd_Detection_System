"""
Main entry point for Crowd Safety Detection System
Run this script to process UCSD sequences
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

from config import Config
from main_detector import CrowdSafetyDetector, process_multiple_sequences
from frame_loader import get_all_sequences, preview_sequence

def main():
    parser = argparse.ArgumentParser(
        description='AI-Based Public Safety Monitoring System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single sequence
  python run_detection.py --sequence datasets/UCSD_Anomaly_Dataset/UCSDped1/Test/Test001
  
  # Process all test sequences for Ped1
  python run_detection.py --dataset ped1 --split test
  
  # Process all sequences in batch
  python run_detection.py --batch --dataset ped1
  
  # Preview a sequence
  python run_detection.py --preview datasets/UCSD_Anomaly_Dataset/UCSDped1/Test/Test001
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--sequence', type=str,
                           help='Path to single sequence folder')
    mode_group.add_argument('--dataset', type=str, choices=['ped1', 'ped2'],
                           help='Process entire dataset (ped1 or ped2)')
    mode_group.add_argument('--preview', type=str,
                           help='Preview a sequence without processing')
    
    # Additional options
    parser.add_argument('--split', type=str, choices=['train', 'test'], default='test',
                       help='Dataset split (default: test)')
    parser.add_argument('--batch', action='store_true',
                       help='Process multiple sequences in batch mode')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable real-time visualization')
    parser.add_argument('--no-save', action='store_true',
                       help='Disable saving outputs')
    
    args = parser.parse_args()
    
    # Update config based on arguments
    if args.no_display:
        Config.DISPLAY_REAL_TIME = False
    
    if args.no_save:
        Config.SAVE_VISUALIZATIONS = False
        Config.SAVE_ALERTS = False
    
    # Execute based on mode
    if args.preview:
        print("\n🔍 PREVIEW MODE")
        preview_sequence(args.preview, num_frames=100)
    
    elif args.sequence:
        print("\n🚀 SINGLE SEQUENCE MODE")
        detector = CrowdSafetyDetector(args.sequence)
        detector.process_sequence()
    
    elif args.dataset:
        dataset_name = f"UCSDped{args.dataset[-1]}"  # ped1 -> UCSDped1
        dataset_path = Path(Config.DATASET_ROOT) / dataset_name
        
        split_name = args.split.capitalize()  # test -> Test
        
        print(f"\n🚀 DATASET MODE: {dataset_name} - {split_name}")
        
        try:
            sequences = get_all_sequences(str(dataset_path), split_name)
            
            if not sequences:
                print(f"❌ No sequences found in {dataset_path}/{split_name}")
                return
            
            print(f"Found {len(sequences)} sequences")
            
            if args.batch:
                process_multiple_sequences(sequences)
            else:
                # Process first sequence as example
                print(f"\nProcessing first sequence: {Path(sequences[0]).name}")
                print("(Use --batch to process all sequences)")
                
                detector = CrowdSafetyDetector(sequences[0])
                detector.process_sequence()
        
        except Exception as e:
            print(f"❌ Error: {e}")
            return


if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║     AI-BASED PUBLIC SAFETY MONITORING SYSTEM                      ║
║     Crowd Behavior Anomaly Detection                              ║
║     Optimized for UCSD Anomaly Detection Dataset                  ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
""")
    
    main()
