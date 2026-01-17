# AI-Based Public Safety Monitoring System

Crowd Behavior Anomaly Detection for UCSD Dataset

See project README inside the file for usage and instructions. This repository contains a CPU-friendly, explainable rule-based system using optical flow and background subtraction to flag crowd safety risks. Run `python quick_test.py` first to verify your environment and dataset placement.
I'll add a short todo plan for running the project, then give concise run steps and commands.

Created 5 todos

Progress: Plan added to the todo list. Next — environment and run commands.

Run Steps

Read docs: See README.md and requirements.txt.
Create env & install: In PowerShell run:
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

Configure paths: Edit the dataset/path settings in src/config.py so it points to your dataset folder (e.g., the datasets/UCSD_Anomaly_Dataset directory).

Quick test: Run the basic smoke test:
python quick_test.py

Explore runner: Check available options then run detection:
python run_detection.py --help
# Example (adjust args per --help):
python run_detection.py --sequence Test001

View outputs: Results, visualizations, and alerts are in outputs/ (see outputs/results, outputs/visualizations, outputs/alerts).

See Image Over Here 
![Sign-Up Page](https://drive.google.com/drive/folders/1wrfMnHIXUFUz7SRTpjK_jQisoibCDf37)
