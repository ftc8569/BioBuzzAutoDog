# BioBuzzAutoDog

FTC vision work: finding game balls with a camera, tracking them in field coordinates, and driving the robot to chase them like a dog. It started in DECODE (2025-2026) with green and purple balls. The default is now BIOBUZZ (2026-2027): yellow POLLEN and red and blue NECTAR, seen through the robot's Limelight 3A.

<img src="https://github.com/user-attachments/assets/fefaeac9-8e91-4be8-9d23-9d82a193d1e3" alt="Ball detection screen recording" width="800">

<img src="https://github.com/user-attachments/assets/1b8c6aac-84fd-48d8-ba3c-4ead6370b630" alt="Ball tracking screen recording" width="800">

## What's here

### `docs/`

- `ftc-auto-dog.pdf` — design document for the auto dog: pixel to field projection, per-track Kalman filter, gating, Mahalanobis association, and pose time sync.

### `python/autodog/` — the auto dog tracker

An implementation of the design doc. It runs off the robot controller: on a Pi next to the camera, or on a laptop reading the Limelight stream (see [BIOBUZZ](#biobuzz-2026-2027-and-the-limelight)):

| Module | Design doc | What it does |
| --- | --- | --- |
| `geometry.py` | 2.1–2.4 | Frames, `T_RC`/`T_FR`, pixel → field projection onto the ball-center plane, Jacobian measurement covariance |
| `gating.py` | 2.5 | Field bounds, ramp polygons (with margin), robot footprint |
| `kalman.py` | 2.6 | Constant-velocity Kalman filter per track (Joseph-form update) |
| `association.py` | 2.7 | Mahalanobis-gated greedy assignment, optional Hungarian |
| `timesync.py` | 2.8 | Pose ring buffer with wrap-aware interpolation, ping-pong clock offset |
| `tracker.py` | 2.9 | The per-frame tick and target selection |
| `link.py` | 2.8 | UDP wire format and Pi side of the link (poses in, targets out) |
| `extrinsics.py` | 2.10 | Fit the camera mount from balls at known field positions |
| `sim.py`, `render.py` | — | Simulated field/robot/camera, and a top-down debug view |

`python/autodog_run.py` drives it:

```
python autodog_run.py sim                      # simulated dog chasing balls (q quits, p pauses)
python autodog_run.py live --height 0.30 --pitch 25 --x 0.3 --y 1.83   # webcam at a fixed spot
python autodog_run.py mount --x 0.61 --y 1.83 --balls "1.83,1.83; 2.44,1.22; 2.44,2.44"
python autodog_run.py robot --rc 192.168.43.1 --pitch 28 --height 0.28  # on the Pi
```

Field frame (as in the doc): origin at a field corner, meters, heading counterclockwise from +x. `TrackerConfig` (in `config.py`) holds every tunable; ramp polygons are empty by default, so measure them for your field and pass a config JSON with `--config`.

#### Where this differs from the design doc

Each change came out of the simulator or the consistency checks in `tests/`:

- **Pose noise in `R_meas`.** The doc propagates pixel noise only. A 0.3° heading error moves a ball 2 m away by 1 cm, which pixel noise does not cover, so far detections failed the χ² gate. `pose_sigma_xy` and `pose_sigma_theta` add it. Set both to 0 for the doc's behavior.
- **Pose extrapolation.** For frames newer than the last pose, the buffer extrapolates along the last two samples (up to 50 ms) rather than holding the last pose. Holding it lagged a robot spinning at 0.9 rad/s by up to a pose period, and the mean Mahalanobis d² on correct matches went from 2 to 4.8.
- **Tentative tracks die fast.** Unconfirmed tracks are deleted after more than 2 consecutive misses (`tentative_max_frames_unseen`). With the doc's 15 frames, a coasting tentative track's gate grows to meters and absorbs random false positives: 38% of them became confirmed ghost tracks. Now it's 2%.
- **Eaten balls.** Tracks inside the robot footprint are deleted (`drop_tracks_under_robot`). Otherwise an intaked ball stays the chase target for another half second.
- **Target hysteresis.** Keeps the current target unless another one is closer by more than `target_switch_margin` (0.2 m), so the dog doesn't dither between two balls.
- `S` uses each detection's own `R_meas` (the doc writes `S_i`), and the update uses the Joseph form.
- `robot` mode re-syncs clocks every 10 s, because crystals drift about 3 ms per minute. It keeps the old offset if a re-sync is noisy.

### `python/` — original vision scripts

Desktop computer vision tests, from [FTCDecodeVision](https://github.com/tanayvin1216/FTCDecodeVision).

- `vision.py` — main AprilTag-anchored green/purple ball detection pipeline
- `vision_fixed.py`, `vision_debug.py`, `vision2.py` — variants and debug views of the pipeline
- `ballDetect.py` — ball detection plus tracking with an OpenCV tracker
- `SimpleBallTracker.py` — `SimpleBallTracker` class: per-frame green/purple detection plus CSRT tracking, meant for robot integration (the autodog runner uses its `detect()`)
- `LuxonisTracking.py` — 3D ball trajectory tracking with a Luxonis OAK-D (depthai 3.x), with CSV logging
- `empiricalTrajectory.py` — empirical trajectory recorder
- `remoteTestingRoom.py` — saves frames and debug masks for offline analysis
- `capture_calib.py`, `calibrate.py`, `calib_imgs/` — chessboard capture and camera intrinsics calibration. `calibrate.py` writes `camera_calib.json`, which `autodog_run.py` loads.

### `robot/`

On-robot Kotlin code from the `tanayVision` branch of [FTC-SigmaCorns-22377/Decode](https://github.com/FTC-SigmaCorns-22377/Decode/tree/tanayVision), kept at its original `TeamCode` paths. These files depend on the rest of that repo and do not build on their own.

- `GlobalShutterVision/eyebrain.kt` — `EyeBrain`, the `VisionProcessor` port of `vision.py` for the global shutter camera
- `opmode/test/VisionDriveTest.kt` — teleop that uses `EyeBrain` on the `globalShutter` webcam to turn toward and chase a ball
- `opmode/test/ShooterExperimentTest.kt` — shooter experiment teleop for camera system integration
- `control/localization/VisionTracker.kt`, `constants/Limelight.kt`, `res/xml/teamwebcamcalibrations.xml` — team localization and camera calibration code the vision work relies on (written by Sigma Corns teammates)

Auto dog, robot-controller side:

- `autodog/AutoDogLink.kt` — the RC half of the UDP link: streams odometry, answers clock-sync pings, receives targets. Plain Kotlin and `java.net`, no FTC dependencies.
- `autodog/ChaseController.kt` — turns toward a field-frame target and drives onto it with the intake (same control law as the simulator)
- `opmode/test/AutoDogChaseTest.kt` — teleop: hold A to chase the Pi's target with the intake on, A+B to also spin and search, release for manual drive
- `src/test/kotlin/sigmacorns/test/AutoDogTest.kt` — JUnit 5 tests for the controller, the frame conversion and a UDP loopback

The team's odometry uses a center-origin field frame; `FieldFrame` converts to the tracker's corner origin.

### BIOBUZZ (2026-2027) and the Limelight

This season's robot code is [ftc8569/2026-BioBuzz](https://github.com/ftc8569/2026-BioBuzz). The local branch `autodog-limelight` there adds the "Auto Dog Chase" TeleOp and `docs/autodog.md`. The robot has a Limelight 3A and no Pi, so `autodog_run.py` runs on a laptop on the robot's Wi-Fi and reads the Limelight stream (`--source limelight`, which is `http://192.168.43.1:5800`).

- `--game biobuzz` is the default. It detects yellow POLLEN (2.8 in) and red and blue NECTAR (3.6 in), reported as colors `Y`, `R` and `B`. Each color is projected at its own center height (`TrackerConfig.biobuzz()`, `ball_radii`). `--game decode` keeps the green/purple pipeline.
- `autodog/detect.py` is an HSV blob detector with a roundness check, so alliance tape doesn't count as a ball. Its thresholds are untuned starting points.
- `autodog/stream.py` keeps only the newest frame of a network stream, so a slow loop doesn't fall behind.
- `autodog/calib.py` calibrates intrinsics from any source: `python -m autodog.calib capture --source limelight`, then `python -m autodog.calib solve`, which writes `limelight_calib.json` for `--calib`. Until then the Limelight uses nominal intrinsics (640x480, 54.5 x 42 degrees).
- `--log DIR` writes `ticks.jsonl` (detections, tracks and targets per frame) plus sampled raw frames, for offline review.
- `--no-field-gate` keeps detections outside the 12 ft field, for testing on a floor where the odometry origin isn't a field corner.
- UDP ports moved from 5800/5801 to 47800 (tracker) and 47801 (RC), because the Control Hub forwards 5800 to 5809 to the Limelight.

## Running

```
cd python
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # add -r requirements-dev.txt for tests
python vision.py                           # original detector on camera 0
python autodog_run.py sim                  # auto dog in simulation
```

## Tests

```
cd python
pip install -r requirements-dev.txt
python -m pytest            # 92 tests, about 10 s, no camera needed
ruff check .                # lint for autodog/, tests/, autodog_run.py
```

The suite includes geometry round trips, a Monte Carlo check of the projection covariance, filter consistency, the UDP link over loopback, closed-loop chase simulations, and every camera script run end to end on a fake camera (`tests/fakecam.py`).

On the robot side, `AutoDogTest.kt` runs with the team repo's normal `gradlew test`.

## Hardware-to-do

The simulator and tests assume the camera model is right. Before trusting the tracker on a field:

1. Calibrate intrinsics for the actual camera and resolution: `python -m autodog.calib capture` then `solve` for the Limelight or any stream (`capture_calib.py` and `calibrate.py` only handle the original webcam on camera 0).
2. Measure the mount, then refine it with `autodog_run.py mount`, using balls on tile corners.
3. Measure `pixel_sigma` from detections of a still ball, and set the ramp polygons and robot footprint in a config JSON. For BIOBUZZ, also tune the HSV ranges in `autodog/detect.py` under the real lights.
4. Measure `--capture-latency`: the time from shutter to `cap.read()` returning.
