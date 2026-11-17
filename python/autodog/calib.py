"""Chessboard intrinsics calibration for any frame source, including the Limelight.

    python -m autodog.calib capture --source limelight --out calib_limelight
    python -m autodog.calib solve calib_limelight --out limelight_calib.json

`capture_calib.py` and `calibrate.py` do the same for the original webcam, but
they are fixed to camera 0 at 1280x720 and rotate every image by 180 degrees.
The JSON written here has the same keys, so `autodog_run.py --calib` loads it.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import cv2
import numpy as np

from .stream import LIMELIGHT_STREAM, LatestFrameReader, resolve_source

PATTERN = (9, 6)       # inner corners of a 10x7-square board
SQUARE_MM = 21.59      # the team's printed board: 0.85 in squares


def find_corners(gray: np.ndarray, pattern: tuple[int, int] = PATTERN) -> np.ndarray | None:
    ok, corners = cv2.findChessboardCorners(gray, pattern, None)
    if not ok:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)


def solve(images: list[np.ndarray], pattern: tuple[int, int] = PATTERN,
          square_mm: float = SQUARE_MM) -> dict:
    """Calibrate from BGR or gray images. Raises ValueError when too few boards are found."""
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * (square_mm / 1000.0)
    objpoints, imgpoints, size = [], [], None
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        if size is None:
            size = (gray.shape[1], gray.shape[0])
        elif size != (gray.shape[1], gray.shape[0]):
            raise ValueError(f"mixed image sizes: {size} and {(gray.shape[1], gray.shape[0])}")
        corners = find_corners(gray, pattern)
        if corners is not None:
            objpoints.append(objp)
            imgpoints.append(corners)
    if len(objpoints) < 5:
        raise ValueError(f"chessboard found in only {len(objpoints)} of {len(images)} images; "
                         "need at least 5 (10 to 25 varied poses is better)")
    rms, K, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, size, None, None)
    return {
        "K": [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])],
        "DIST_COEFFS": [float(x) for x in dist.ravel()[:5]],
        "IMAGE_WIDTH_PX": size[0], "IMAGE_HEIGHT_PX": size[1],
        "RMS_PX": float(rms),
        "BOARDS_USED": len(objpoints),
    }


def run_capture(args) -> int:
    os.makedirs(args.out, exist_ok=True)
    count = len(glob.glob(os.path.join(args.out, "img_*.jpg")))
    reader = LatestFrameReader(resolve_source(args.source))
    print(f"{args.source}: {reader.width}x{reader.height}")
    print("SPACE = save frame   |   q = quit")
    try:
        while True:
            ok, frame, _ = reader.read()
            if not ok:
                print("stream stopped")
                return 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(
                gray, PATTERN, None, cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_ADAPTIVE_THRESH)
            preview = frame.copy()
            if found:
                cv2.drawChessboardCorners(preview, PATTERN, corners, found)
            cv2.putText(preview, f"{count}/{args.target}   detected: {'YES' if found else 'no'}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0) if found else (0, 0, 255), 2)
            cv2.imshow("calib capture", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                count += 1
                path = os.path.join(args.out, f"img_{count:02d}.jpg")
                cv2.imwrite(path, frame)
                print(f"saved {path}   (chessboard {'detected' if found else 'NOT detected'})")
    finally:
        reader.release()
        cv2.destroyAllWindows()
    print(f"{count} images in {args.out}/")
    return 0


def run_solve(args) -> int:
    files = sorted(glob.glob(os.path.join(args.dir, "*.jpg")) +
                   glob.glob(os.path.join(args.dir, "*.png")))
    if not files:
        raise SystemExit(f"no images in {args.dir}/")
    images = [cv2.imread(f) for f in files]
    if args.rotate_180:
        images = [cv2.rotate(i, cv2.ROTATE_180) for i in images]
    try:
        calib = solve(images, square_mm=args.square_mm)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    rms = calib["RMS_PX"]
    quality = ("great" if rms < 0.5 else "acceptable" if rms < 1.0 else
               "marginal" if rms < 1.5 else "RECAPTURE with more varied board poses")
    print(f"{calib['BOARDS_USED']} of {len(files)} images used, "
          f"{calib['IMAGE_WIDTH_PX']}x{calib['IMAGE_HEIGHT_PX']}")
    print(f"RMS reprojection error: {rms:.3f} px  [{quality}]")
    fx, fy, cx, cy = calib["K"]
    print(f"fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")
    with open(args.out, "w") as fh:
        json.dump(calib, fh, indent=2)
    print(f"wrote {args.out}; pass it to autodog_run.py with --calib")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m autodog.calib", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture", help="save chessboard frames from a camera or stream")
    c.add_argument("--source", default="limelight",
                   help=f"camera index, stream URL, or 'limelight' ({LIMELIGHT_STREAM})")
    c.add_argument("--out", default="calib_limelight", help="directory for img_NN.jpg")
    c.add_argument("--target", type=int, default=25, help="how many images to aim for")
    c.set_defaults(fn=run_capture)
    s = sub.add_parser("solve", help="compute intrinsics from saved frames")
    s.add_argument("dir", nargs="?", default="calib_limelight")
    s.add_argument("--out", default="limelight_calib.json")
    s.add_argument("--square-mm", type=float, default=SQUARE_MM)
    s.add_argument("--rotate-180", action="store_true",
                   help="images were taken with the camera upside down")
    s.set_defaults(fn=run_solve)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
