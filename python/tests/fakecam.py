"""Run a camera script end to end with a fake camera and headless windows.

usage: python tests/fakecam.py <script.py> [frames] [script args...]

cv2.VideoCapture yields 1280x720 frames with a green and a purple ball that
slide sideways; imshow/trackbars/mouse callbacks become no-ops; waitKey
returns 'q' every `frames` calls so each display loop exits. Set
FAKECAM_SAVE=path.png to keep the last image a script showed, and
FAKECAM_IMAGE=path.png to serve that image as every frame instead.
"""
import builtins
import os
import runpy
import sys

import cv2
import numpy as np

SCRIPT = os.path.abspath(sys.argv[1])
FRAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 40
W, H = 1280, 720


def frame(i):
    img = np.full((H, W, 3), (90, 90, 90), np.uint8)
    dx = (i * 4) % 200
    # green ball left, purple ball right (sizes inside the 300..20000 px area gate)
    cv2.circle(img, (400 + dx, 360), 45, (60, 170, 30), -1)
    cv2.circle(img, (800 + dx, 360), 45, (150, 50, 140), -1)
    return img


_STILL = cv2.imread(os.environ["FAKECAM_IMAGE"]) if os.environ.get("FAKECAM_IMAGE") else None


class FakeCap:
    def __init__(self, *a, **k):
        self.i = 0

    def isOpened(self):
        return True

    def read(self):
        self.i += 1
        return True, (_STILL.copy() if _STILL is not None else frame(self.i))

    def set(self, *a):
        return True

    def get(self, prop):
        return {cv2.CAP_PROP_FRAME_WIDTH: W, cv2.CAP_PROP_FRAME_HEIGHT: H,
                cv2.CAP_PROP_FPS: 30.0}.get(prop, 0.0)

    def release(self):
        pass


calls = {"n": 0, "shown": set()}


def waitKey(_d=0):
    calls["n"] += 1
    if calls["n"] % FRAMES == 0:
        return ord("q")
    return 0xFF


def imshow(name, img):
    calls["shown"].add(name)
    if os.environ.get("FAKECAM_SAVE"):
        cv2.imwrite(os.environ["FAKECAM_SAVE"], img)


cv2.VideoCapture = FakeCap
cv2.imshow = imshow
cv2.waitKey = waitKey
cv2.namedWindow = lambda *a, **k: None
cv2.setMouseCallback = lambda *a, **k: None
cv2.destroyAllWindows = lambda *a, **k: None
cv2.destroyWindow = lambda *a, **k: None
cv2.createTrackbar = lambda *a, **k: None
cv2.getTrackbarPos = lambda name, win: 0
cv2.setTrackbarPos = lambda *a, **k: None
cv2.selectROI = lambda *a, **k: (380, 290, 140, 140)

builtins.input = lambda *a: ""

sys.path.insert(0, os.path.dirname(SCRIPT))
os.chdir(os.path.dirname(SCRIPT))
sys.argv = [SCRIPT] + sys.argv[3:]
try:
    runpy.run_path(SCRIPT, run_name="__main__")
    status = "OK"
except SystemExit as e:
    status = f"SystemExit({e.code})"
except KeyboardInterrupt:
    status = "stopped"
print(f"\n[fakecam] {os.path.basename(SCRIPT)}: {status}; "
      f"waitKey calls={calls['n']}; windows={sorted(calls['shown'])}")
