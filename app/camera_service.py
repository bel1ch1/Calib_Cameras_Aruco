import threading
import time
from typing import Dict, Generator

import cv2

try:
    from picamera2 import Picamera2
except ImportError:  # pragma: no cover - unavailable on non-RPi dev machines
    Picamera2 = None


class CameraError(RuntimeError):
    """Raised when camera initialization or frame capture fails."""


class CameraService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cameras: Dict[int, "Picamera2"] = {}

    def _ensure_camera(self, camera_id: int) -> "Picamera2":
        if Picamera2 is None:
            raise CameraError("picamera2 is not installed on this system.")

        with self._lock:
            if camera_id in self._cameras:
                return self._cameras[camera_id]

            cam = Picamera2(camera_id)
            config = cam.create_preview_configuration(main={"size": (1280, 720), "format": "RGB888"})
            cam.configure(config)
            cam.start()
            time.sleep(0.5)
            self._cameras[camera_id] = cam
            return cam

    def capture_frame(self, camera_id: int):
        cam = self._ensure_camera(camera_id)
        frame_rgb = cam.capture_array()
        if frame_rgb is None:
            raise CameraError(f"No frame received from camera {camera_id}.")
        # Keep RGB as the canonical output format from the camera service.
        return frame_rgb

    def mjpeg_generator(self, camera_id: int) -> Generator[bytes, None, None]:
        while True:
            frame_rgb = self.capture_frame(camera_id)
            # OpenCV JPEG encoder expects BGR channel order.
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            ok, encoded = cv2.imencode(".jpg", frame_bgr)
            if not ok:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"
            )
            time.sleep(0.03)

    def close_all(self) -> None:
        with self._lock:
            cams = list(self._cameras.values())
            self._cameras.clear()

        for cam in cams:
            try:
                cam.stop()
                cam.close()
            except Exception:
                pass

