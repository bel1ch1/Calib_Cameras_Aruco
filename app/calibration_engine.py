import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .config import MIN_IMAGES_FOR_CALIBRATION, OUTPUT_DIR


def _as_serializable_array(arr: np.ndarray) -> List[List[float]]:
    return np.asarray(arr).tolist()


def _load_images(snapshot_dir: Path) -> List[Path]:
    return sorted(snapshot_dir.glob("*.png"))


def _detect_checkerboard_corners(gray: np.ndarray, checkerboard_size: Tuple[int, int]):
    """
    Robust checkerboard detection with multiple OpenCV methods and pre-processing fallbacks.
    Returns (ret, corners).
    """
    base_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    fast_flags = base_flags | cv2.CALIB_CB_FAST_CHECK

    # Try modern SB detector first if available (often much more robust).
    if hasattr(cv2, "findChessboardCornersSB"):
        try:
            ret, corners = cv2.findChessboardCornersSB(
                gray,
                checkerboard_size,
                flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            if ret:
                return ret, corners
        except Exception:
            pass

    # Classic detector on original image.
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, base_flags)
    if ret:
        return ret, corners

    # Fallbacks for difficult lighting/contrast cases.
    gray_eq = cv2.equalizeHist(gray)
    ret, corners = cv2.findChessboardCorners(gray_eq, checkerboard_size, base_flags)
    if ret:
        return ret, corners

    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, corners = cv2.findChessboardCorners(gray_blur, checkerboard_size, base_flags)
    if ret:
        return ret, corners

    gray_inv = cv2.bitwise_not(gray)
    ret, corners = cv2.findChessboardCorners(gray_inv, checkerboard_size, base_flags)
    if ret:
        return ret, corners

    # Last quick check to avoid full failure on borderline frames.
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, fast_flags)
    return ret, corners


def _save_outputs(camera_id: int, payload: Dict, raw: Dict) -> Dict[str, str]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / f"calibration_cam{camera_id}.json"
    npz_path = OUTPUT_DIR / f"calibration_cam{camera_id}.npz"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    np.savez(
        npz_path,
        ret=raw["ret"],
        cameraMatrix=raw["camera_matrix"],
        distCoeffs=raw["dist_coeffs"],
        rvecs=raw["rvecs"],
        tvecs=raw["tvecs"],
    )
    return {"json": str(json_path), "npz": str(npz_path)}


def calibrate_checkerboard(
    *,
    camera_id: int,
    snapshot_dir: Path,
    checkerboard_size: Tuple[int, int],
    square_size_m: float,
) -> Dict:
    image_paths = _load_images(snapshot_dir)
    if len(image_paths) < MIN_IMAGES_FOR_CALIBRATION:
        raise ValueError(
            f"At least {MIN_IMAGES_FOR_CALIBRATION} snapshots are required, got {len(image_paths)}."
        )

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((1, checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0 : checkerboard_size[0], 0 : checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size_m

    objpoints = []
    imgpoints = []
    image_size = None

    detected_files = []
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]
        ret, corners = _detect_checkerboard_corners(gray, checkerboard_size)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            detected_files.append(path.name)

    if len(objpoints) < MIN_IMAGES_FOR_CALIBRATION:
        raise ValueError(
            "Not enough valid checkerboard detections. "
            f"Need at least {MIN_IMAGES_FOR_CALIBRATION}, got {len(objpoints)}."
        )

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    payload = {
        "camera_id": camera_id,
        "target_type": "checkerboard",
        "valid_images": len(objpoints),
        "total_images": len(image_paths),
        "detected_files": detected_files,
        "reprojection_error": float(ret),
        "camera_matrix": _as_serializable_array(camera_matrix),
        "dist_coeffs": _as_serializable_array(dist_coeffs),
    }
    files = _save_outputs(
        camera_id,
        payload,
        {
            "ret": ret,
            "camera_matrix": camera_matrix,
            "dist_coeffs": dist_coeffs,
            "rvecs": rvecs,
            "tvecs": tvecs,
        },
    )
    payload["files"] = files
    return payload


def _get_aruco_dictionary(dictionary_name: str):
    dictionary_id = getattr(cv2.aruco, dictionary_name, None)
    if dictionary_id is None:
        raise ValueError(f"Unknown ArUco dictionary: {dictionary_name}")
    return cv2.aruco.getPredefinedDictionary(dictionary_id)


def _build_charuco_board(squares_x: int, squares_y: int, square_len: float, marker_len: float, dictionary):
    if hasattr(cv2.aruco, "CharucoBoard"):
        return cv2.aruco.CharucoBoard((squares_x, squares_y), square_len, marker_len, dictionary)
    if hasattr(cv2.aruco, "CharucoBoard_create"):
        return cv2.aruco.CharucoBoard_create(squares_x, squares_y, square_len, marker_len, dictionary)
    raise RuntimeError("OpenCV build does not support ChArUco board creation.")


def calibrate_charuco(
    *,
    camera_id: int,
    snapshot_dir: Path,
    squares_x: int,
    squares_y: int,
    square_length: float,
    marker_length: float,
    dictionary_name: str,
) -> Dict:
    image_paths = _load_images(snapshot_dir)
    if len(image_paths) < MIN_IMAGES_FOR_CALIBRATION:
        raise ValueError(
            f"At least {MIN_IMAGES_FOR_CALIBRATION} snapshots are required, got {len(image_paths)}."
        )

    dictionary = _get_aruco_dictionary(dictionary_name)
    board = _build_charuco_board(squares_x, squares_y, square_length, marker_length, dictionary)

    if hasattr(cv2.aruco, "DetectorParameters"):
        detector_params = cv2.aruco.DetectorParameters()
    else:
        detector_params = cv2.aruco.DetectorParameters_create()

    all_charuco_corners = []
    all_charuco_ids = []
    image_size = None

    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]

        if hasattr(cv2.aruco, "ArucoDetector"):
            detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
            corners, ids, _ = detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=detector_params)

        if ids is None or len(ids) == 0:
            continue

        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray,
            board=board,
        )
        if charuco_ids is not None and len(charuco_ids) >= 4:
            all_charuco_corners.append(charuco_corners)
            all_charuco_ids.append(charuco_ids)

    if len(all_charuco_corners) < MIN_IMAGES_FOR_CALIBRATION:
        raise ValueError(
            "Not enough valid ChArUco detections. "
            f"Need at least {MIN_IMAGES_FOR_CALIBRATION}, got {len(all_charuco_corners)}."
        )

    if not hasattr(cv2.aruco, "calibrateCameraCharuco"):
        raise RuntimeError("OpenCV build does not support calibrateCameraCharuco.")

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=all_charuco_corners,
        charucoIds=all_charuco_ids,
        board=board,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None,
    )

    payload = {
        "camera_id": camera_id,
        "target_type": "charuco",
        "valid_images": len(all_charuco_corners),
        "total_images": len(image_paths),
        "reprojection_error": float(ret),
        "camera_matrix": _as_serializable_array(camera_matrix),
        "dist_coeffs": _as_serializable_array(dist_coeffs),
    }
    files = _save_outputs(
        camera_id,
        payload,
        {
            "ret": ret,
            "camera_matrix": camera_matrix,
            "dist_coeffs": dist_coeffs,
            "rvecs": rvecs,
            "tvecs": tvecs,
        },
    )
    payload["files"] = files
    return payload

