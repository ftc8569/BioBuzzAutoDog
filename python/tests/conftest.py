import math
import sys
from pathlib import Path

import pytest

PYTHON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_DIR))

from autodog import DEFAULT_INTRINSICS, CameraMount, TrackerConfig  # noqa: E402


@pytest.fixture
def intrinsics():
    return DEFAULT_INTRINSICS


@pytest.fixture
def mount():
    return CameraMount(x=0.15, y=0.0, z=0.28, pitch_rad=math.radians(28.0))


@pytest.fixture
def cfg():
    return TrackerConfig()
