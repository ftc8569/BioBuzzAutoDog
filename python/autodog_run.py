"""Run the auto dog tracker: in simulation, on a webcam, or on the robot.

  python autodog_run.py sim                  # watch the simulated dog chase balls
  python autodog_run.py sim --headless --seconds 20 --save out/
  python autodog_run.py live --height 0.30 --pitch 25
  python autodog_run.py robot --rc 192.168.43.1 --pitch 28 --height 0.28
  python autodog_run.py robot --rc 192.168.43.1 --source limelight --show --log runs/
  python autodog_run.py mount --x 0.61 --y 1.83 --balls "1.83,1.83; 2.44,1.22; 2.44,2.44"

Keys in the windows: q quits, p pauses (sim).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # find autodog/ even under python -I or from another cwd

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from autodog import BallTracker, CameraIntrinsics, CameraMount, Detection, TrackerConfig  # noqa: E402
from autodog.config import DEFAULT_INTRINSICS, LIMELIGHT_3A_INTRINSICS  # noqa: E402
from autodog.render import FieldView, draw_camera, synth_camera_image  # noqa: E402
from autodog.stream import LIMELIGHT_STREAM, LatestFrameReader, resolve_source  # noqa: E402


def load_intrinsics(path: str | None, width: int, height: int,
                    source: int | str = 0) -> CameraIntrinsics:
    if path:
        intr = CameraIntrinsics.from_calibration(path)
    elif source == LIMELIGHT_STREAM:
        # camera_calib.json is for a different (wide) webcam, not the Limelight.
        intr = LIMELIGHT_3A_INTRINSICS
    elif (HERE / "camera_calib.json").exists():
        intr = CameraIntrinsics.from_calibration(HERE / "camera_calib.json")
    else:
        intr = DEFAULT_INTRINSICS
    if (intr.width, intr.height) != (width, height):
        intr = intr.scaled(width, height)
    return intr


def load_config(path: str | None, game: str = "decode", field_gate: bool = True) -> TrackerConfig:
    if path:
        cfg = TrackerConfig.from_json(Path(path).read_text())
    else:
        cfg = TrackerConfig.biobuzz() if game == "biobuzz" else TrackerConfig()
    if not field_gate:
        cfg.gate_field = False
    return cfg


def config_from_args(args) -> TrackerConfig:
    return load_config(args.config, args.game, not args.no_field_gate)


def side_by_side(left: np.ndarray, right: np.ndarray, height: int = 540) -> np.ndarray:
    def fit(img):
        s = height / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * s), height))
    return np.hstack([fit(left), fit(right)])


# --------------------------------------------------------------------------- sim
def run_sim(args) -> int:
    from autodog.sim import SimConfig, Simulation

    sim = Simulation(chase=not args.no_chase, sim_cfg=SimConfig(seed=args.seed))
    view = FieldView(sim.tracker)
    save = Path(args.save) if args.save else None
    if save:
        save.mkdir(parents=True, exist_ok=True)
    paused = False
    frame_no = 0
    last = None
    while sim.t < args.seconds:
        if not paused:
            entry = None
            while entry is None and sim.t < args.seconds:
                entry = sim.step()
            if entry is None:
                break
            last = entry
            frame_no += 1
        if last is None:
            continue
        field = view.draw(last.result, truth=last.balls, robot_pose=sim.robot.pose)
        cam = draw_camera(synth_camera_image(sim), last.detections, last.result)
        canvas = side_by_side(field, cam)
        if save and frame_no % args.save_every == 0:
            cv2.imwrite(str(save / f"frame_{frame_no:05d}.png"), canvas)
        if not args.headless:
            cv2.imshow("autodog sim", canvas)
            key = cv2.waitKey(1 if not paused else 30) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p"):
                paused = not paused
    floor = [b for b in sim.balls if b.z_offset == 0]
    print(f"simulated {sim.t:.1f} s, {len(sim.log)} frames")
    print(f"collected {len(sim.collected)}/{len(floor)} floor balls: " +
          ", ".join(f"{b.color} at {t:.1f}s" for t, b in sim.collected))
    errs = sim.track_errors()
    if errs:
        print("confirmed track errors (cm): " + " ".join(f"{100 * e:.1f}" for e in errs))
    if not args.headless:
        cv2.destroyAllWindows()
    return 0


# -------------------------------------------------------------------- detection
class PixelDetector:
    """Pixel detections for the game: DECODE green/purple, or BIOBUZZ Y/R/B."""

    def __init__(self, rotate_180: bool = False, game: str = "decode") -> None:
        self.rotate_180 = rotate_180
        self.game = game
        if game == "biobuzz":
            from autodog.detect import ColorBallDetector

            self.impl = ColorBallDetector()
        else:
            sys.path.insert(0, str(HERE))
            from SimpleBallTracker import SimpleBallTracker

            self.impl = SimpleBallTracker()

    def __call__(self, frame: np.ndarray) -> tuple[np.ndarray, list[Detection]]:
        if self.rotate_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        if self.game == "biobuzz":
            return frame, self.impl(frame)
        dets = []
        for b in self.impl.detect(frame):
            x, y, w, h = b["bbox"]
            # bbox center ~ ball center, which is what lies on the z = r plane.
            # A box over pixel columns x..x+w-1 is centered at x + (w - 1) / 2.
            dets.append(Detection(x + (w - 1) / 2.0, y + (h - 1) / 2.0, b["color"]))
        return frame, dets


def mount_from_args(args) -> CameraMount:
    return CameraMount(
        x=args.cam_x, y=args.cam_y, z=args.height,
        pitch_rad=math.radians(args.pitch), yaw_rad=math.radians(args.yaw),
        roll_rad=math.radians(args.roll),
    )


def open_source(args) -> tuple[LatestFrameReader, int | str]:
    """--source (camera index, URL or 'limelight'), else --camera."""
    source = resolve_source(args.source) if args.source else args.camera
    reader = LatestFrameReader(source, args.width, args.height_px)
    return reader, source


def open_log(args, intr, mount, cfg, source):
    if not getattr(args, "log", None):
        return None
    from autodog.runlog import RunLog

    log = RunLog(args.log, {"args": vars(args), "source": str(source), "intrinsics": intr,
                            "mount": mount, "config": cfg}, frame_every=args.log_every)
    print(f"logging to {log.dir}")
    return log


# -------------------------------------------------------------------------- live
def run_live(args) -> int:
    """Webcam at a fixed, known spot: shows balls on the field map."""
    cap, source = open_source(args)
    w, h = cap.width, cap.height
    intr = load_intrinsics(args.calib, w, h, source)
    cfg = config_from_args(args)
    mount = mount_from_args(args)
    tracker = BallTracker(intr, mount, cfg)
    pose = (args.x, args.y, math.radians(args.heading))
    detector = PixelDetector(args.rotate_180, args.game)
    view = FieldView(tracker)
    log = open_log(args, intr, mount, cfg, source)
    print(f"camera {w}x{h}, fx={intr.fx:.1f}; pose {pose}; q to quit")
    while True:
        ok, frame, t = cap.read()
        if not ok:
            break
        frame, dets = detector(frame)
        result = tracker.tick(t, dets, pose)
        if log:
            log.tick(frame, t, dets, result, tracker, result.target)
        canvas = side_by_side(view.draw(result, robot_pose=pose),
                              draw_camera(frame, dets, result))
        cv2.imshow("autodog live", canvas)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    if log:
        log.close()
    cv2.destroyAllWindows()
    return 0


# ------------------------------------------------------------------------- robot
def run_robot(args) -> int:
    """On the Pi: camera -> detector -> tracker -> UDP target to the RC."""
    from autodog.link import PiLink

    cap, source = open_source(args)
    w, h = cap.width, cap.height
    intr = load_intrinsics(args.calib, w, h, source)
    cfg = config_from_args(args)
    mount = mount_from_args(args)
    link = PiLink((args.rc, args.rc_port), bind=("0.0.0.0", args.port))
    tracker = BallTracker(intr, mount, cfg, poses=link.poses)
    detector = PixelDetector(args.rotate_180, args.game)
    log = open_log(args, intr, mount, cfg, source)
    print(f"camera {source} {w}x{h}, fx={intr.fx:.1f}")
    print(f"syncing clocks with {args.rc}:{args.rc_port} ...")
    sync = link.sync_clocks()
    print(f"offset {sync.offset:+.6f} s, std {1e3 * sync.offset_std:.2f} ms, "
          f"rtt {1e3 * sync.round_trip:.2f} ms")
    if not sync.ok:
        print("WARNING: clock offset std above 5 ms; poses may be skewed")
    view = FieldView(tracker) if args.show else None
    frames, t_report = 0, time.monotonic()
    t_sync = t_report
    try:
        while True:
            # Pi and RC crystals drift ~50 ppm (~3 ms/min): re-sync regularly.
            if args.resync > 0 and time.monotonic() - t_sync > args.resync:
                t_sync = time.monotonic()
                try:
                    old = link.sync.offset
                    new = link.sync_clocks(timeout=0.05)
                    if not new.ok:
                        print(f"WARNING: clock re-sync std {1e3 * new.offset_std:.2f} ms")
                    elif abs(new.offset - old) > 0.002:
                        print(f"clock offset moved {1e3 * (new.offset - old):+.1f} ms")
                except TimeoutError:
                    print("WARNING: RC did not answer clock re-sync; keeping old offset")
            ok, frame, t_arrive = cap.read()
            # One capture time per frame. The frame arrives after exposure,
            # encoding and (for the Limelight) Wi-Fi, so back it off by the
            # measured shutter-to-arrival delay.
            t_frame = t_arrive - args.capture_latency
            if not ok:
                print("camera stopped")
                break
            link.poll()
            frame, dets = detector(frame)
            result = tracker.tick(t_frame, dets)
            now = time.monotonic()
            link.poll()
            # Hand the controller the state predicted to *now* (doc 2.10).
            target = tracker.target(now)
            link.send_target(now, target)
            if log:
                log.tick(frame, t_frame, dets, result, tracker, target,
                         {"t_sent": round(now, 4), "poses": len(link.poses)})
            frames += 1
            if now - t_report > 2.0:
                tgt = tracker.target(now)
                print(f"{frames / (now - t_report):5.1f} fps  dropped {cap.dropped}  "
                      f"poses {len(link.poses)}  "
                      f"tracks {len(tracker.tracks)}  target "
                      + ("none" if tgt is None else
                         f"#{tgt.track_id} {tgt.color} ({tgt.position[0]:.2f}, {tgt.position[1]:.2f})"))
                frames, t_report = 0, now
            if view is not None:
                cv2.imshow("autodog robot", side_by_side(view.draw(result),
                                                         draw_camera(frame, dets, result)))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        link.close()
        if log:
            log.close()
    return 0


# ------------------------------------------------------------------------- mount
def parse_balls(text: str) -> list[tuple[float, float]]:
    """'1.22,1.83; 1.83,1.22' -> [(1.22, 1.83), (1.83, 1.22)]"""
    out = []
    for chunk in text.replace(" ", "").split(";"):
        if chunk:
            x, y = chunk.split(",")
            out.append((float(x), float(y)))
    return out


def run_mount(args) -> int:
    """Fit the camera mount from balls placed at known field positions."""
    from autodog.extrinsics import field_errors, fit_mount_from_detections

    balls = parse_balls(args.balls)
    cap, source = open_source(args)
    w, h = cap.width, cap.height
    intr = load_intrinsics(args.calib, w, h, source)
    cfg = config_from_args(args)
    guess = mount_from_args(args)
    pose = (args.x, args.y, math.radians(args.heading))
    detector = PixelDetector(args.rotate_180, args.game)
    frames = []
    for _ in range(args.frames):
        ok, frame, _ = cap.read()
        if not ok:
            break
        frame, dets = detector(frame)
        frames.append([(d.u, d.v) for d in dets])
    cap.release()
    try:
        fit, obs = fit_mount_from_detections(
            intr, guess, cfg.ball_radius, pose, frames, balls, args.match_px)
    except ValueError:
        print("fewer than 3 known balls matched a detection; check --balls, the pose, "
              "or raise --match-px")
        return 2
    seen = {o.ball_xy for o in obs}
    print(f"{len(obs)} detections matched to {len(seen)}/{len(balls)} balls over {len(frames)} frames")
    before = field_errors(intr, guess, cfg.ball_radius, obs)
    print(f"current mount: worst field error {100 * before.max():.1f} cm, "
          f"mean {100 * before.mean():.1f} cm")
    print(f"fitted mount:  {fit.describe()}")
    if fit.rms_px > 3.0:
        print("WARNING: RMS above 3 px; a ball may be mismatched, occluded or misplaced")
    m = fit.mount
    pitch, yaw, roll = (math.degrees(a) for a in (m.pitch_rad, m.yaw_rad, m.roll_rad))
    print(f"use: --cam-x {m.x:.3f} --cam-y {m.y:.3f} --height {m.z:.3f} "
          f"--pitch {pitch:.2f} --yaw {yaw:.2f} --roll {roll:.2f}")
    return 0


# -------------------------------------------------------------------------- main
def add_camera_args(p) -> None:
    p.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    p.add_argument("--source", help="camera index, stream URL, or 'limelight' "
                   f"({LIMELIGHT_STREAM}); overrides --camera")
    p.add_argument("--game", choices=("biobuzz", "decode"), default="biobuzz",
                   help="ball colors and sizes (default: this season, biobuzz)")
    p.add_argument("--no-field-gate", action="store_true",
                   help="keep detections outside the 12 ft field (testing off-field)")
    p.add_argument("--log", help="directory for run logs (ticks.jsonl + frames)")
    p.add_argument("--log-every", type=int, default=15, help="save every Nth raw frame")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height-px", type=int, default=720)
    p.add_argument("--calib", help="camera_calib.json from calibrate.py")
    p.add_argument("--config", help="TrackerConfig JSON (field, ramps, noise)")
    p.add_argument("--rotate-180", action="store_true",
                   help="camera mounted upside down (calib_imgs were)")
    p.add_argument("--cam-x", type=float, default=0.15, help="camera ahead of robot center (m)")
    p.add_argument("--cam-y", type=float, default=0.0, help="camera left of robot center (m)")
    p.add_argument("--height", type=float, default=0.28, help="camera height above floor (m)")
    p.add_argument("--pitch", type=float, default=28.0, help="degrees down from level")
    p.add_argument("--yaw", type=float, default=0.0, help="degrees left")
    p.add_argument("--roll", type=float, default=0.0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    s = sub.add_parser("sim", help="simulated field, camera and robot")
    s.add_argument("--seconds", type=float, default=30.0)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--no-chase", action="store_true", help="robot holds still")
    s.add_argument("--headless", action="store_true")
    s.add_argument("--save", help="directory for PNG frames")
    s.add_argument("--save-every", type=int, default=15)

    lv = sub.add_parser("live", help="webcam at a fixed spot on the field")
    add_camera_args(lv)
    lv.add_argument("--x", type=float, default=0.3, help="robot x on the field (m)")
    lv.add_argument("--y", type=float, default=1.83, help="robot y on the field (m)")
    lv.add_argument("--heading", type=float, default=0.0, help="degrees from +x")

    rb = sub.add_parser("robot", help="Pi on the robot, talking to the RC over UDP")
    add_camera_args(rb)
    from autodog.link import DEFAULT_PI_PORT, DEFAULT_RC_PORT
    rb.add_argument("--rc", required=True, help="robot controller IP")
    rb.add_argument("--rc-port", type=int, default=DEFAULT_RC_PORT)
    rb.add_argument("--port", type=int, default=DEFAULT_PI_PORT)
    rb.add_argument("--capture-latency", type=float, default=0.0,
                    help="seconds from shutter to cap.read() returning")
    rb.add_argument("--show", action="store_true", help="open debug windows")
    rb.add_argument("--resync", type=float, default=10.0,
                    help="seconds between clock re-syncs (0 = only at start)")

    mt = sub.add_parser("mount", help="fit camera pitch/roll/yaw/height from balls at known spots")
    add_camera_args(mt)
    mt.add_argument("--balls", required=True,
                    help="known ball centers on the field, 'x,y; x,y; ...' in meters "
                         "(tile corners are multiples of 0.6096)")
    mt.add_argument("--x", type=float, required=True, help="robot x on the field (m)")
    mt.add_argument("--y", type=float, required=True, help="robot y on the field (m)")
    mt.add_argument("--heading", type=float, default=0.0, help="degrees from +x")
    mt.add_argument("--frames", type=int, default=30)
    mt.add_argument("--match-px", type=float, default=120.0,
                    help="max pixel distance between a predicted and detected ball")

    args = ap.parse_args(argv)
    modes = {"sim": run_sim, "live": run_live, "robot": run_robot, "mount": run_mount}
    return modes[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
