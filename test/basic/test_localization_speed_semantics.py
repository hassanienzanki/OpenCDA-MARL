"""Localization speed semantics: 3D magnitude via misc.get_speed()."""

from pathlib import Path
import sys

_TEST_DIR = str(Path(__file__).resolve().parent)
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)

from test_helpers import load_module, StubLocation, StubVelocity, StubVehicle

misc_module = load_module("misc", "opencda/core/common/misc.py")


def test_get_speed_returns_3d_magnitude():
    vehicle = StubVehicle(StubLocation(), StubVelocity(3.0, 4.0, 12.0))

    speed_kmh = misc_module.get_speed(vehicle)
    speed_mps = misc_module.get_speed(vehicle, meters=True)

    # 3D magnitude: sqrt(9+16+144) = 13 m/s = 46.8 km/h
    assert round(speed_kmh, 2) == 46.8
    assert round(speed_mps, 2) == 13.0


if __name__ == "__main__":
    test_get_speed_returns_3d_magnitude()
    print("test_get_speed_returns_3d_magnitude: ok")
