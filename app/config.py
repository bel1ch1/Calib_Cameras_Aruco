from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
STATIC_DIR = ROOT_DIR / "static"

DEFAULT_CHECKERBOARD_SIZE = (7, 7)
DEFAULT_SQUARE_SIZE_METERS = 0.041

DEFAULT_CHARUCO = {
    "squares_x": 5,
    "squares_y": 7,
    "square_length": 0.04,
    "marker_length": 0.02,
    "dictionary": "DICT_4X4_50",
}

MIN_IMAGES_FOR_CALIBRATION = 6

