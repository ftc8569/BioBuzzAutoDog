"""autodog.calib: recover known intrinsics from rendered chessboard views."""

import json

import cv2
import numpy as np
import pytest

from autodog import calib
from autodog.geometry import CameraIntrinsics

W, H = 640, 480
FX, FY, CX, CY = 620.0, 615.0, 322.0, 236.0
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1.0]])
SQ = calib.SQUARE_MM / 1000.0


def board_texture(px: int = 40) -> tuple[np.ndarray, float]:
    """10x7 squares with a one-square white border; returns the image and px per meter."""
    cols, rows = calib.PATTERN[0] + 1, calib.PATTERN[1] + 1
    img = np.full(((rows + 2) * px, (cols + 2) * px), 255, np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                img[(r + 1) * px:(r + 2) * px, (c + 1) * px:(c + 2) * px] = 0
    return img, px / SQ


def render(rvec, tvec) -> np.ndarray:
    """View of the board: first inner corner at the board-frame origin, as in calib.solve."""
    tex, ppm = board_texture()
    h, w = tex.shape
    # Texture pixel -> board meters: the first inner corner sits two squares in.
    corners_px = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    corners_m = np.column_stack([corners_px / ppm - 2 * SQ, np.zeros(4)]).astype(np.float32)
    img_pts, _ = cv2.projectPoints(corners_m, np.array(rvec, float), np.array(tvec, float), K, None)
    Hm = cv2.getPerspectiveTransform(corners_px, img_pts.reshape(4, 2).astype(np.float32))
    return cv2.warpPerspective(tex, Hm, (W, H), flags=cv2.INTER_AREA, borderValue=128)


POSES = [
    ((0.0, 0.0, 0.0), (-0.10, -0.07, 0.35)),
    ((0.4, 0.0, 0.0), (-0.10, -0.05, 0.40)),
    ((-0.4, 0.1, 0.0), (-0.08, -0.08, 0.42)),
    ((0.0, 0.5, 0.0), (-0.09, -0.06, 0.40)),
    ((0.0, -0.5, 0.1), (-0.11, -0.06, 0.45)),
    ((0.3, 0.3, 0.2), (-0.06, -0.08, 0.45)),
    ((-0.3, -0.3, -0.2), (-0.12, -0.04, 0.45)),
    ((0.2, -0.4, 0.0), (-0.14, -0.07, 0.50)),
    ((0.5, 0.2, -0.1), (-0.05, -0.05, 0.50)),
]


def test_solve_recovers_intrinsics():
    result = calib.solve([render(r, t) for r, t in POSES])
    fx, fy, cx, cy = result["K"]
    assert result["BOARDS_USED"] == len(POSES)
    assert (result["IMAGE_WIDTH_PX"], result["IMAGE_HEIGHT_PX"]) == (W, H)
    assert result["RMS_PX"] < 0.5
    assert fx == pytest.approx(FX, rel=0.02)
    assert fy == pytest.approx(FY, rel=0.02)
    assert cx == pytest.approx(CX, abs=8)
    assert cy == pytest.approx(CY, abs=8)


def test_solve_needs_enough_boards():
    blank = np.full((H, W), 128, np.uint8)
    with pytest.raises(ValueError, match="only 1 of 4"):
        calib.solve([render(*POSES[0]), blank, blank, blank])


def test_solve_cli_writes_json_autodog_can_load(tmp_path):
    for i, (r, t) in enumerate(POSES):
        cv2.imwrite(str(tmp_path / f"img_{i:02d}.png"), render(r, t))
    out = tmp_path / "calib.json"
    assert calib.main(["solve", str(tmp_path), "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    intr = CameraIntrinsics.from_calibration(out)
    assert intr.fx == pytest.approx(data["K"][0])
    assert (intr.width, intr.height) == (W, H)
