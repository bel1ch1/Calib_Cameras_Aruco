import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from .config import DATA_DIR, DEFAULT_CHARUCO, DEFAULT_CHECKERBOARD_SIZE, DEFAULT_SQUARE_SIZE_METERS


@dataclass
class CameraSession:
    camera_id: int
    target_type: str = "checkerboard"
    checkerboard_rows: int = DEFAULT_CHECKERBOARD_SIZE[0]
    checkerboard_cols: int = DEFAULT_CHECKERBOARD_SIZE[1]
    square_size_m: float = DEFAULT_SQUARE_SIZE_METERS
    charuco_squares_x: int = DEFAULT_CHARUCO["squares_x"]
    charuco_squares_y: int = DEFAULT_CHARUCO["squares_y"]
    charuco_square_length: float = DEFAULT_CHARUCO["square_length"]
    charuco_marker_length: float = DEFAULT_CHARUCO["marker_length"]
    charuco_dictionary: str = DEFAULT_CHARUCO["dictionary"]
    status: str = "idle"
    message: str = ""
    last_result: Optional[dict] = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["snapshot_count"] = self.snapshot_count
        return data

    @property
    def snapshot_dir(self) -> Path:
        return DATA_DIR / f"cam{self.camera_id}"

    @property
    def snapshot_count(self) -> int:
        if not self.snapshot_dir.exists():
            return 0
        return len(list(self.snapshot_dir.glob("*.png")))


class SessionStore:
    def __init__(self) -> None:
        self._sessions: Dict[int, CameraSession] = {}

    def get_or_create(self, camera_id: int) -> CameraSession:
        if camera_id not in self._sessions:
            self._sessions[camera_id] = CameraSession(camera_id=camera_id)
        session = self._sessions[camera_id]
        session.snapshot_dir.mkdir(parents=True, exist_ok=True)
        return session

    def update_settings(self, camera_id: int, payload: dict) -> CameraSession:
        session = self.get_or_create(camera_id)
        for key, value in payload.items():
            if value is None:
                continue
            if hasattr(session, key):
                setattr(session, key, value)
        return session

    def save_snapshot(self, camera_id: int, frame: np.ndarray) -> Path:
        session = self.get_or_create(camera_id)
        session.snapshot_dir.mkdir(parents=True, exist_ok=True)
        index = session.snapshot_count + 1
        file_path = session.snapshot_dir / f"snapshot_{index:04d}.png"
        if not cv2.imwrite(str(file_path), frame):
            raise RuntimeError("Failed to save snapshot to disk.")
        session.status = "capturing"
        session.message = f"Snapshot saved: {file_path.name}"
        return file_path

    def clear_snapshots(self, camera_id: int) -> None:
        session = self.get_or_create(camera_id)
        if session.snapshot_dir.exists():
            shutil.rmtree(session.snapshot_dir)
        session.snapshot_dir.mkdir(parents=True, exist_ok=True)
        session.status = "idle"
        session.message = "All snapshots deleted."

