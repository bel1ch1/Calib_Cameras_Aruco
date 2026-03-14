# Dual-Camera Calibration Web App

This project is a web application for calibrating two Raspberry Pi CSI cameras (camera `0` and `1`) and generating camera parameters for OpenCV ArUco marker tasks.

The app is designed for:
- Raspberry Pi 5
- Ubuntu Server (headless, no desktop GUI)
- Picamera2 camera access (`picamera v2.1` devices)

Main output per camera:
- `camera_matrix`
- `dist_coeffs` (distortion coefficients)

These values are required for accurate ArUco detection, pose estimation, and orientation workflows.

## Features

- Live web preview for each camera using Picamera2
- Snapshot capture during calibration
- Snapshot counter and status messages
- Delete all snapshots, cancel session, finish calibration
- Supports checkerboard and ChArUco calibration targets
- Outputs camera matrix and distortion coefficients
- Saves calibration results as JSON and NPZ

## Project structure

- `app/main.py` - FastAPI app and API routes
- `app/camera_service.py` - Picamera2 frame capture and MJPEG stream
- `app/session_store.py` - per-camera calibration session and snapshot handling
- `app/calibration_engine.py` - checkerboard and ChArUco calibration
- `static/index.html` - browser UI
- `data/` - captured snapshots
- `output/` - generated calibration result files

## Setup on Raspberry Pi 5 (Ubuntu Server)

1. Install system dependencies and Python:
   - `sudo apt update`
   - `sudo apt install -y python3 python3-pip libatlas-base-dev libopenblas-dev libjpeg-dev`
2. (Recommended) create virtual environment:
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
3. Install Python dependencies:
   - `pip install -r requirements.txt`

## Run

- Start server:
  - `uvicorn app.main:app --host 0.0.0.0 --port 8000`
- Open browser from another machine on same network:
  - `http://<raspberry-pi-ip>:8000`

## Typical workflow

For each camera panel (`Camera 0` and `Camera 1`):

1. Select calibration target (`checkerboard` or `charuco`).
2. Click `Start`.
3. Move the board to different positions/angles in front of the camera.
4. Click `Capture snapshot` multiple times.
5. Check that snapshot counter increases and status messages show successful saves.
6. If needed, click `Delete all snapshots` to restart cleanly.
7. Click `Finish calibration` to compute calibration parameters automatically.
8. Copy results from the UI or download output files.

## Calibration usage

1. Select calibration target (Checkerboard or ChArUco) for a camera panel.
2. Set board size in `n x n` format and physical sizes:
   - Checkerboard: `n x n` means printed squares count; OpenCV automatically converts this to inner corners `(n-1) x (n-1)`.
   - ChArUco: set board `n x n`, square size, and marker size.
3. Click `Start`.
4. Position calibration board in multiple poses.
5. Click `Capture snapshot` repeatedly (at least 6 usable images).
6. Click `Finish calibration`.
7. Copy matrix/distortion from UI, or download JSON/NPZ files.

## Notes

- Checkerboard default follows `newCalibrationParameters.py`: `7x7` inner corners, `0.041 m` square size.
- For a standard chessboard with `8x8` printed squares, use `7x7` inner corners in OpenCV (already set as default).
- The UI accepts board dimensions in `n x n` format (examples: `8x8`, `6x6`).
- Ensure both cameras are available as Picamera2 camera indices `0` and `1`.

## Output files

After successful calibration, the app writes:

- `output/calibration_cam0.json`
- `output/calibration_cam0.npz`
- `output/calibration_cam1.json`
- `output/calibration_cam1.npz`

`JSON` is convenient for reading/copying parameters in applications, while `NPZ` is convenient for direct NumPy/OpenCV loading.

