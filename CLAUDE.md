# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

BioBuzzAutoDog (`ftc8569/BioBuzzAutoDog`): FTC DECODE (2025-2026) vision work.
Detect green and purple balls, track them in field coordinates, and drive the
robot to chase them. `docs/ftc-auto-dog.pdf` is the design;
`python/autodog/` implements it. README.md maps every file and lists where the
implementation intentionally departs from the doc (and why).

## Python (`python/`)

- Environment: `python/.venv` (gitignored).
  `cd python && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt -r requirements-dev.txt`
- Tests: `.venv/bin/python -m pytest` (about 10 s, no camera). Lint:
  `.venv/bin/ruff check .`; `pyproject.toml` limits ruff to `autodog/`,
  `tests/` and `autodog_run.py`. The original scripts are deliberately unlinted.
- Use `opencv-contrib-python`, not `opencv-python`: `ballDetect.py` and
  `SimpleBallTracker.py` need `cv2.TrackerCSRT_create`.
- `tests/fakecam.py` runs any camera script headless: synthetic frames,
  no-op windows, `q` after N frames. `FAKECAM_IMAGE` serves a fixed image and
  `FAKECAM_SAVE` keeps the last shown frame. Use it rather than a real camera
  for anything automated.
- `vision.py` is `# %%` cells for the VS Code interactive window. Top-level
  calls must sit under `if __name__ == "__main__":`, because
  `remoteTestingRoom.py` imports it.
- Run scripts from `python/`: paths like `calib_imgs/` and
  `camera_calib.json` are relative to it.

### autodog conventions

- Field frame: origin at a field corner, meters, CCW heading. Robot frame:
  x forward, y left. Camera frame: OpenCV (x right, y down, z forward).
  `CameraMount.pitch_rad > 0` tilts the camera toward the floor.
- Every tunable lives in `TrackerConfig` (`autodog/config.py`). Doc defaults
  are kept; the deviations are commented there and listed in the README.
- Changing tracker behavior: check filter consistency (NEES or mean d² on
  true matches, about 2) and ghost-track rates with `autodog/sim.py`, not only
  final error. Those caught the real bugs here.
- Wire format (`autodog/link.py`) is mirrored byte for byte in
  `robot/.../autodog/AutoDogLink.kt`. Change both, and keep the size
  assertions in `tests/test_link.py` and `AutoDogTest.kt` in step.

## Robot (`robot/`)

Kotlin at its original `TeamCode` paths from `FTC-SigmaCorns-22377/Decode`
branch `tanayVision` (a clone lives at `~/Desktop/SigmaCorns-Decode`). The
`autodog/` package plus `AutoDogChaseTest.kt` and `AutoDogTest.kt` are new;
copy them into that repo to build.

- `AutoDogLink.kt` and `ChaseController.kt` have no FTC dependencies. Compile
  and test them standalone with Android Studio's bundled compiler
  (`java -cp .../plugins/Kotlin/kotlinc/lib/kotlin-compiler.jar
  org.jetbrains.kotlin.cli.jvm.K2JVMCompiler`, using the bundled JBR 21) and the
  JUnit console launcher.
- The full Gradle build needs Java 21 (Android Studio's JBR; the system Java
  25 is too new for Gradle 8.13) and a `local.properties` with `sdk.dir`,
  `username` and `apiKey`. The last is a GitHub token with `read:packages`
  for the Sloth plugin on GitHub Packages.
- Team odometry is center-origin; convert with `FieldFrame`.

### BIOBUZZ robot (current season)

- Robot code: `ftc8569/2026-BioBuzz`, cloned at `~/Desktop/RoboKnights/2026-BioBuzz`,
  local branch `autodog-limelight` (not pushed). Its copies of `AutoDogLink.kt` and
  `ChaseController.kt` live in `TeamCode/.../teamcode/autodog/`; keep them in step
  with `robot/` here and with `link.py`.
- That repo's `AGENTS.md` forbids installing, deploying or running OpModes on the
  robot unless the user explicitly revokes it. Local builds and tests are fine:
  `JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home ./gradlew :TeamCode:testDebugUnitTest`
  (it needs JDK 17; JBR 21 fails the toolchain check).
