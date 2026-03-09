from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from .calibration_engine import calibrate_charuco, calibrate_checkerboard
from .camera_service import CameraError, CameraService
from .config import STATIC_DIR
from .session_store import SessionStore

app = FastAPI(title="Dual-Camera Calibration App")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

camera_service = CameraService()
session_store = SessionStore()


def _validate_camera_id(camera_id: int) -> None:
    if camera_id not in (0, 1):
        raise HTTPException(status_code=400, detail="Only camera 0 and 1 are supported.")


class SessionSettings(BaseModel):
    target_type: Literal["checkerboard", "charuco"] = "checkerboard"
    checkerboard_rows: int = Field(default=7, ge=2)
    checkerboard_cols: int = Field(default=7, ge=2)
    square_size_m: float = Field(default=0.041, gt=0)
    charuco_squares_x: int = Field(default=5, ge=3)
    charuco_squares_y: int = Field(default=7, ge=3)
    charuco_square_length: float = Field(default=0.04, gt=0)
    charuco_marker_length: float = Field(default=0.02, gt=0)
    charuco_dictionary: str = "DICT_4X4_50"


class FinishRequest(BaseModel):
    settings: Optional[SessionSettings] = None


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    index_path = STATIC_DIR / "index.html"
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/stream/{camera_id}")
def stream(camera_id: int) -> StreamingResponse:
    _validate_camera_id(camera_id)
    try:
        generator = camera_service.mjpeg_generator(camera_id)
        return StreamingResponse(generator, media_type="multipart/x-mixed-replace; boundary=frame")
    except CameraError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/sessions/{camera_id}")
def get_session(camera_id: int) -> dict:
    _validate_camera_id(camera_id)
    session = session_store.get_or_create(camera_id)
    return session.to_dict()


@app.post("/sessions/{camera_id}/start")
def start_session(camera_id: int, settings: SessionSettings) -> dict:
    _validate_camera_id(camera_id)
    session = session_store.update_settings(camera_id, settings.model_dump())
    session.status = "capturing"
    session.message = "Calibration session started."
    return session.to_dict()


@app.post("/sessions/{camera_id}/capture")
def capture_snapshot(camera_id: int) -> dict:
    _validate_camera_id(camera_id)
    try:
        frame = camera_service.capture_frame(camera_id)
        path = session_store.save_snapshot(camera_id, frame)
        session = session_store.get_or_create(camera_id)
        return {
            "ok": True,
            "message": f"Snapshot saved: {path.name}",
            "snapshot_count": session.snapshot_count,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/sessions/{camera_id}/delete-all")
def delete_all_snapshots(camera_id: int) -> dict:
    _validate_camera_id(camera_id)
    session_store.clear_snapshots(camera_id)
    return {"ok": True, "message": "All snapshots deleted.", "snapshot_count": 0}


@app.post("/sessions/{camera_id}/cancel")
def cancel_session(camera_id: int) -> dict:
    _validate_camera_id(camera_id)
    session = session_store.get_or_create(camera_id)
    session.status = "cancelled"
    session.message = "Calibration cancelled by user."
    return {"ok": True, "message": session.message}


@app.post("/sessions/{camera_id}/finish")
def finish_calibration(camera_id: int, req: FinishRequest) -> dict:
    _validate_camera_id(camera_id)
    session = session_store.get_or_create(camera_id)
    if req.settings is not None:
        session = session_store.update_settings(camera_id, req.settings.model_dump())

    try:
        if session.target_type == "checkerboard":
            result = calibrate_checkerboard(
                camera_id=camera_id,
                snapshot_dir=session.snapshot_dir,
                checkerboard_size=(session.checkerboard_rows, session.checkerboard_cols),
                square_size_m=session.square_size_m,
            )
        elif session.target_type == "charuco":
            result = calibrate_charuco(
                camera_id=camera_id,
                snapshot_dir=session.snapshot_dir,
                squares_x=session.charuco_squares_x,
                squares_y=session.charuco_squares_y,
                square_length=session.charuco_square_length,
                marker_length=session.charuco_marker_length,
                dictionary_name=session.charuco_dictionary,
            )
        else:
            raise ValueError(f"Unsupported target_type: {session.target_type}")

        session.status = "completed"
        session.message = "Calibration finished successfully."
        session.last_result = result
        return {"ok": True, "message": session.message, "result": result}
    except Exception as exc:
        session.status = "error"
        session.message = str(exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/sessions/{camera_id}/result")
def get_result(camera_id: int) -> dict:
    _validate_camera_id(camera_id)
    session = session_store.get_or_create(camera_id)
    if not session.last_result:
        raise HTTPException(status_code=404, detail="No calibration result available.")
    return session.last_result


@app.get("/download/{camera_id}/{file_kind}")
def download_result(camera_id: int, file_kind: Literal["json", "npz"]) -> FileResponse:
    _validate_camera_id(camera_id)
    session = session_store.get_or_create(camera_id)
    if not session.last_result:
        raise HTTPException(status_code=404, detail="No calibration result available.")

    file_path = Path(session.last_result["files"][file_kind])
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(path=file_path, filename=file_path.name)


@app.on_event("shutdown")
def shutdown_event() -> None:
    camera_service.close_all()

