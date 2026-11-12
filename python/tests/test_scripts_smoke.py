"""Every camera script runs end to end on a fake camera with no windows.

Each script runs in its own subprocess through tests/fakecam.py, which swaps
cv2.VideoCapture for synthetic frames and makes the display calls no-ops.
"""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

PY = Path(__file__).resolve().parents[1]
FAKECAM = PY / "tests" / "fakecam.py"

CAMERA_SCRIPTS = [
    "vision.py",
    "vision_fixed.py",
    "vision_debug.py",
    "ballDetect.py",
    "remoteTestingRoom.py",
    "empiricalTrajectory.py",
    "capture_calib.py",
]


def run(script, frames=30, *args, timeout=90, env=None):
    proc = subprocess.run(
        [sys.executable, "-I", str(FAKECAM), script, str(frames), *args],
        cwd=PY, capture_output=True, text=True, timeout=timeout,
        env={**os.environ, **(env or {})},
    )
    tail = (proc.stdout + proc.stderr)[-3000:]
    assert proc.returncode == 0, tail
    status = [ln for ln in proc.stdout.splitlines() if ln.startswith("[fakecam]")]
    assert status, tail
    return status[-1], proc.stdout


@pytest.fixture
def no_new_files():
    """Scripts that log CSVs or snapshots must not leave files behind."""
    before = set(PY.rglob("*"))
    yield
    for path in set(PY.rglob("*")) - before:
        if path.is_file() and "__pycache__" not in path.parts and ".pytest_cache" not in path.parts:
            path.unlink()


@pytest.mark.parametrize("script", CAMERA_SCRIPTS)
def test_camera_script_runs(script, no_new_files):
    status, _ = run(script)
    assert ": OK;" in status or "SystemExit(0)" in status, status
    assert "windows=[]" not in status  # it actually displayed frames


def test_calibrate_writes_intrinsics(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-I", "calibrate.py"], cwd=PY, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "[GREAT]" in proc.stdout
    assert (PY / "camera_calib.json").exists()


def test_autodog_sim_headless():
    proc = subprocess.run(
        [sys.executable, "-I", "autodog_run.py", "sim", "--headless", "--seconds", "20"],
        cwd=PY, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "collected 5/5 floor balls" in proc.stdout, proc.stdout


def test_autodog_live_mode_tracks_fake_balls(tmp_path):
    shot = tmp_path / "live.png"
    status, out = run("autodog_run.py", 40, "live", "--game", "decode", env={"FAKECAM_SAVE": str(shot)})
    assert "autodog live" in status
    assert shot.exists() and shot.stat().st_size > 10_000


def test_autodog_robot_mode_talks_to_rc():
    sys.path.insert(0, str(PY))
    from autodog.link import FakeRC

    rc = FakeRC(("127.0.0.1", 0), offset_s=50.0)
    stop = threading.Event()

    def serve():
        while not stop.is_set():
            # stream a pose so the Pi can project frames, answer pings
            rc.send_pose(0.3, 1.83, 0.0)
            rc.step()
            time.sleep(0.01)

    # the Pi binds a fixed port; tell FakeRC where it will be
    import socket

    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    pi_port = probe.getsockname()[1]
    probe.close()
    rc.pi_address = ("127.0.0.1", pi_port)
    th = threading.Thread(target=serve, daemon=True)
    th.start()
    try:
        status, out = run(
            "autodog_run.py", 60, "robot", "--game", "decode", "--rc", "127.0.0.1",
            "--rc-port", str(rc.address[1]), "--port", str(pi_port), "--show",
        )
    finally:
        stop.set()
        th.join(1)
        rc.close()
    assert "offset +50.0" in out, out
    assert rc.targets, "RC never received a target"
    assert any(t.has_target for t in rc.targets)


def test_autodog_mount_fit_recovers_true_mount(tmp_path):
    """Render balls at tile corners with a known mount; fit from a bad guess."""
    import math

    import cv2
    import numpy as np

    sys.path.insert(0, str(PY))
    from autodog import DEFAULT_INTRINSICS, CameraMount
    from autodog.geometry import Projector

    true = CameraMount(x=0.15, y=0.0, z=0.26, pitch_rad=math.radians(30), roll_rad=math.radians(1.0))
    pose = (0.61, 1.83, 0.0)
    balls = [(1.22, 1.22), (1.22, 2.44), (1.83, 1.83), (1.83, 1.22), (1.83, 2.44), (2.44, 1.83)]
    r = 0.0635
    cam = Projector(DEFAULT_INTRINSICS, true, r, 1e9)
    cam.set_pose(*pose)
    img = np.full((720, 1280, 3), (90, 90, 90), np.uint8)
    for i, (x, y) in enumerate(balls):
        u, v = cam.field_to_pixel([[x, y, r]])[0]
        rad = int(DEFAULT_INTRINSICS.fx * r / math.dist((x, y, r), cam.o_f))
        cv2.circle(img, (round(u), round(v)), rad, (60, 170, 30) if i % 2 else (150, 50, 140), -1)
    still = tmp_path / "balls.png"
    cv2.imwrite(str(still), img)

    _, out = run(
        "autodog_run.py", 5, "mount", "--game", "decode", "--frames", "5",
        "--x", str(pose[0]), "--y", str(pose[1]), "--heading", "0",
        "--balls", "; ".join(f"{x},{y}" for x, y in balls),
        "--height", "0.30", "--pitch", "26", "--cam-x", "0.15",
        env={"FAKECAM_IMAGE": str(still)},
    )
    assert "matched to 6/6 balls" in out, out
    line = next(ln for ln in out.splitlines() if ln.startswith("use: "))
    vals = dict(zip(line.split()[1::2], map(float, line.split()[2::2]), strict=True))
    # one ball is partly hidden behind another; re-pairing still converges
    assert vals["--pitch"] == pytest.approx(30.0, abs=0.2)
    assert vals["--height"] == pytest.approx(0.26, abs=0.005)
    assert vals["--roll"] == pytest.approx(1.0, abs=0.2)
    assert "WARNING" not in out


def test_autodog_biobuzz_limelight_style_run_logs_and_targets(tmp_path):
    """BIOBUZZ robot mode as on the laptop: Limelight-size frames, Y/R balls, --log."""
    import json
    import math
    import socket

    import cv2
    import numpy as np

    sys.path.insert(0, str(PY))
    from autodog import CameraMount
    from autodog.config import INCH, LIMELIGHT_3A_INTRINSICS
    from autodog.geometry import Projector
    from autodog.link import FakeRC

    pose = (0.3, 1.83, 0.0)
    mount = CameraMount(x=0.15, y=0.0, z=0.28, pitch_rad=math.radians(28))
    img = np.full((480, 640, 3), (110, 110, 110), np.uint8)
    for (bx, by), r, bgr in (((1.0, 1.83), 1.4 * INCH, (30, 210, 240)),
                             ((1.3, 1.6), 1.8 * INCH, (40, 40, 210))):
        cam = Projector(LIMELIGHT_3A_INTRINSICS, mount, r, 1e9)
        cam.set_pose(*pose)
        u, v = cam.field_to_pixel(np.array([[bx, by, r]]))[0]
        cv2.circle(img, (int(round(u)), int(round(v))), 22, bgr, -1)
    still = tmp_path / "ll.png"
    cv2.imwrite(str(still), img)
    calib = tmp_path / "ll_calib.json"
    i = LIMELIGHT_3A_INTRINSICS
    calib.write_text(json.dumps({"K": [i.fx, i.fy, i.cx, i.cy], "DIST_COEFFS": i.dist,
                                 "IMAGE_WIDTH_PX": 640, "IMAGE_HEIGHT_PX": 480}))

    rc = FakeRC(("127.0.0.1", 0), offset_s=50.0)
    stop = threading.Event()

    def serve():
        while not stop.is_set():
            rc.send_pose(*pose)
            rc.step()
            time.sleep(0.01)

    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    pi_port = probe.getsockname()[1]
    probe.close()
    rc.pi_address = ("127.0.0.1", pi_port)
    th = threading.Thread(target=serve, daemon=True)
    th.start()
    try:
        status, out = run(
            "autodog_run.py", 60, "robot", "--rc", "127.0.0.1",
            "--rc-port", str(rc.address[1]), "--port", str(pi_port), "--show",
            "--calib", str(calib), "--height", "0.28", "--pitch", "28",
            "--log", str(tmp_path / "runs"), "--log-every", "10",
            env={"FAKECAM_IMAGE": str(still)},
        )
    finally:
        stop.set()
        th.join(1)
        rc.close()
    hits = [t for t in rc.targets if t.has_target]
    assert hits, out
    assert hits[-1].color == "Y", out  # the nearer ball
    assert abs(hits[-1].x - 1.0) < 0.03 and abs(hits[-1].y - 1.83) < 0.03, hits[-1]
    (run_dir,) = (tmp_path / "runs").iterdir()
    ticks = [json.loads(x) for x in (run_dir / "ticks.jsonl").read_text().splitlines()]
    assert ticks and {d[2] for d in ticks[-1]["dets"]} == {"Y", "R"}
    assert list((run_dir / "frames").glob("*.jpg"))
