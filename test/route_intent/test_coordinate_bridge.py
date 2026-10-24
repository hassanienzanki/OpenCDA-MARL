"""The SUMO<->CARLA frame bridge, read from the map rather than hardcoded."""
import pytest

from opencda_marl._kernel.traffic.coordinate_bridge import (
    LEGACY_NET_OFFSET,
    CoordinateBridge,
)

NET_HEADER = """<?xml version="1.0" encoding="UTF-8"?>
<net version="1.16" junctionCornerDetail="5" limitTurnSpeed="5.50">
    <location netOffset="{offset}" convBoundary="0.00,0.00,200.00,200.00"
              origBoundary="-100.00,-100.00,100.00,100.00" projParameter="!"/>
    <edge id="e1" from="a" to="b"/>
</net>
"""


def _net(tmp_path, offset="12.50,-7.25"):
    path = tmp_path / "map.net.xml"
    path.write_text(NET_HEADER.format(offset=offset), encoding="utf-8")
    return path


def test_round_trip_is_exact():
    bridge = CoordinateBridge((99.8, 100.0))
    assert bridge.sumo_to_carla(99.8, 100.0) == (0.0, 0.0)
    assert bridge.carla_to_sumo(0.0, 0.0) == (99.8, 100.0)
    x, y = bridge.carla_to_sumo(*bridge.sumo_to_carla(150.0, -20.0))
    assert (round(x, 6), round(y, 6)) == (150.0, -20.0)


def test_offset_comes_from_the_network_header(tmp_path):
    bridge = CoordinateBridge.from_net_file(_net(tmp_path))
    assert bridge.net_offset == (12.5, -7.25)
    # A different map therefore converts differently — the point of the class.
    assert bridge.sumo_to_carla(12.5, -7.25) == (0.0, 0.0)


def test_missing_file_or_header_falls_back_to_the_legacy_offset(tmp_path):
    assert CoordinateBridge.from_net_file(None).net_offset == LEGACY_NET_OFFSET
    assert CoordinateBridge.from_net_file(tmp_path / "nope.net.xml").net_offset \
        == LEGACY_NET_OFFSET
    bare = tmp_path / "bare.net.xml"
    bare.write_text('<?xml version="1.0"?>\n<net version="1.16"><edge id="e"/></net>\n',
                    encoding="utf-8")
    assert CoordinateBridge.from_net_file(bare).net_offset == LEGACY_NET_OFFSET


def test_unparsable_network_does_not_raise(tmp_path):
    broken = tmp_path / "broken.net.xml"
    broken.write_text("<net><location netOffset=", encoding="utf-8")
    assert CoordinateBridge.from_net_file(broken).net_offset == LEGACY_NET_OFFSET


def test_frames_must_not_be_mixed():
    """The defect this class exists to prevent.

    Subtracting a CARLA-frame centre from a SUMO-frame position leaves the
    whole netOffset in the result — on the bundled map that is ~140 m, so a
    vehicle sitting on the junction still reads as far away.
    """
    bridge = CoordinateBridge((99.8, 100.0))
    junction_sumo = (99.8, 99.57)
    junction_carla = bridge.sumo_to_carla(*junction_sumo)

    mixed = ((junction_sumo[0] - junction_carla[0]) ** 2
             + (junction_sumo[1] - junction_carla[1]) ** 2) ** 0.5
    correct_x, correct_y = bridge.sumo_to_carla(*junction_sumo)
    corrected = ((correct_x - junction_carla[0]) ** 2
                 + (correct_y - junction_carla[1]) ** 2) ** 0.5

    assert mixed == pytest.approx(141.0, abs=1.0)
    assert corrected == pytest.approx(0.0, abs=1e-9)


def test_y_flip_mirrors_only_the_y_axis():
    plain = CoordinateBridge((100.0, 200.0))
    mirrored = CoordinateBridge((100.0, 200.0), y_flip=True)
    assert plain.sumo_to_carla(150.0, 260.0) == (50.0, 60.0)
    assert mirrored.sumo_to_carla(150.0, 260.0) == (50.0, -60.0)
    # Round trip holds under either convention.
    for bridge in (plain, mirrored):
        x, y = bridge.carla_to_sumo(*bridge.sumo_to_carla(311.0, -47.0))
        assert (round(x, 6), round(y, 6)) == (311.0, -47.0)


def test_calibration_picks_the_convention_from_a_landmark():
    """A CARLA town export: the mirrored convention is the one that fits.

    netconvert keeps CARLA's junction ids as node ids, so a junction
    centre is a landmark: here a node sits at SUMO (150, 40) on a network
    with netOffset (100, 100), and CARLA reports the same junction at
    (50, 60). Only the mirrored convention reconciles the two; the
    unmirrored one misses by 120 m.
    """
    bridge = CoordinateBridge((100.0, 100.0))
    assert bridge.calibrate([((150.0, 40.0), (50.0, 60.0))]) is True
    assert bridge.y_flip is True
    got = bridge.sumo_to_carla(150.0, 40.0)
    assert got[0] == pytest.approx(50.0, abs=1e-9)
    assert got[1] == pytest.approx(60.0, abs=1e-9)


def test_calibration_leaves_an_unmirrored_map_alone():
    # The bundled intersection: junction 4 at SUMO (99.8, 99.57) is CARLA
    # (0.0, -0.43) with no mirror.
    bridge = CoordinateBridge((99.80, 100.00))
    assert bridge.calibrate([((99.8, 99.57), (0.0, -0.43))]) is False
    assert bridge.y_flip is False
