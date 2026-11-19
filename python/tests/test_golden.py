"""The golden run that the on-robot Kotlin tracker is checked against."""

from pathlib import Path

from autodog import golden

GOLDEN = Path(__file__).parent / "data" / "tracker_golden.txt"


def test_golden_file_matches_the_tracker():
    """Fails when tracker behavior changes.

    Then regenerate (python -m autodog.golden tests/data/tracker_golden.txt), copy the
    file to 2026-BioBuzz TeamCode/src/test/resources/autodog/, and port the change to
    the Kotlin tracker there until its BallTrackerGoldenTest passes.
    """
    assert golden.record() == GOLDEN.read_text()


def test_golden_run_exercises_the_tracker():
    kinds = [line.split(" ", 1)[0] for line in GOLDEN.read_text().splitlines()]
    assert kinds.count("TICK") > 300
    assert kinds.count("TRK") > 500
    targets = [ln for ln in GOLDEN.read_text().splitlines() if ln.startswith("TGT ")]
    assert sum("none" not in ln for ln in targets) > 100
    res = [ln.split() for ln in GOLDEN.read_text().splitlines() if ln.startswith("RES ")]
    # projection, field and ramp gates all reject something; tracks spawn and die
    for col in (2, 3, 4, 7, 8):
        assert sum(int(r[col]) for r in res) > 0
