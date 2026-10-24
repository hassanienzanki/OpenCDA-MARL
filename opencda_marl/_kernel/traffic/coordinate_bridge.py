"""The one place that knows how SUMO coordinates and CARLA coordinates relate.

A SUMO network produced from a CARLA map is translated by ``netOffset``,
recorded in the ``<location>`` header of the ``.net.xml``. Every conversion
in the framework used the same three hardcoded constants instead, which
tied the code to one specific network: a different map silently shifts
every converted position.

    CARLA = SUMO - netOffset          SUMO = CARLA + netOffset

and, for a network imported from a CARLA *town* export, the y axis is
mirrored as well, because CARLA's world is left-handed while the
OpenDRIVE it exports is not:

    CARLA_y = -(SUMO_y - netOffset_y)

The mirror depends on where the OpenDRIVE came from and is not recorded in
the ``.net.xml`` header, so it is explicit here rather than guessed. The
bundled intersection map is authored in the unmirrored frame — and being
symmetric about its own y axis it would hide a wrong choice — so
``y_flip`` defaults to False and a town map sets it. A landmark settles
which convention a network uses: netconvert keeps CARLA's junction ids as
node ids, so a junction centre known in both frames lands on its node
under the right convention and far from it under the wrong one (see
:meth:`CoordinateBridge.calibrate`).

Reading the offset from the network makes the conversion a property of the
map rather than of the source, and makes "which frame is this number in?"
answerable in one place.
"""
from __future__ import annotations

import math

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Tuple

from loguru import logger

#: The historical constant: the offset of the bundled custom intersection.
#: Used only when a network file cannot be read, so behaviour on that map is
#: unchanged even if the header is missing.
LEGACY_NET_OFFSET: Tuple[float, float] = (99.80, 100.00)


class CoordinateBridge:
    """Convert between a SUMO network's frame and CARLA's."""

    def __init__(self, net_offset: Tuple[float, float] = LEGACY_NET_OFFSET,
                 y_flip: bool = False):
        self.net_offset = (float(net_offset[0]), float(net_offset[1]))
        #: True when the source OpenDRIVE mirrors CARLA's y axis (CARLA
        #: town exports do; the bundled intersection does not).
        self.y_flip = bool(y_flip)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_net_file(cls, net_file: str | Path | None,
                      y_flip: bool = False) -> "CoordinateBridge":
        """Read ``netOffset`` from a ``.net.xml`` header.

        Falls back to :data:`LEGACY_NET_OFFSET` (with a warning) when the
        file is missing or carries no location header, so a run never dies
        over a coordinate lookup.
        """
        if not net_file:
            return cls(y_flip=y_flip)
        path = Path(net_file)
        if not path.is_file():
            logger.warning(f"network file not found for coordinate bridge: {path}")
            return cls(y_flip=y_flip)
        try:
            for _event, element in ET.iterparse(str(path), events=("start",)):
                if element.tag == "location":
                    raw = element.get("netOffset")
                    element.clear()
                    if not raw:
                        break
                    x_str, _, y_str = raw.partition(",")
                    bridge = cls((float(x_str), float(y_str)), y_flip=y_flip)
                    logger.info(f"coordinate bridge: netOffset={bridge.net_offset} "
                                f"from {path.name}")
                    return bridge
                if element.tag == "net":
                    continue
                # The location header is the first child of <net>; stop early
                # rather than parse a multi-megabyte network.
                element.clear()
                break
        except ET.ParseError as exc:
            logger.warning(f"could not parse {path.name} for netOffset: {exc}")
        logger.warning(f"no netOffset in {path.name}; using {LEGACY_NET_OFFSET}")
        return cls(y_flip=y_flip)

    # ------------------------------------------------------------------ #
    def sumo_to_carla(self, x: float, y: float) -> Tuple[float, float]:
        dy = float(y) - self.net_offset[1]
        return (float(x) - self.net_offset[0], -dy if self.y_flip else dy)

    def heading_from_sumo_angle(self, angle_deg: float) -> float:
        """SUMO's compass angle as a heading in this bridge's frame.

        SUMO reports degrees clockwise from north. A heading here is what
        CARLA's is: radians, counter-clockwise from +x, in ``[-pi, pi)``.
        With ``y_flip`` the frame's y axis points the other way, so the
        heading mirrors about the x axis. Used for every SUMO heading and
        velocity that shares a frame with bridged positions; the compass
        angle used raw was 90 degrees off and mirrored, so a velocity built
        from it pointed away from the vehicle's own displacement.
        """
        return sumo_angle_to_heading(angle_deg, y_flip=self.y_flip)

    def carla_to_sumo(self, x: float, y: float) -> Tuple[float, float]:
        dy = -float(y) if self.y_flip else float(y)
        return (float(x) + self.net_offset[0], dy + self.net_offset[1])

    def calibrate(self, samples) -> bool:
        """Pick the convention that matches known CARLA positions.

        ``samples`` is an iterable of ``(sumo_xy, carla_xy)`` pairs. One
        landmark is enough, and netconvert keeps CARLA's junction ids as
        node ids, so a junction centre is the natural landmark. Sets and
        returns :attr:`y_flip`.
        """
        def error(flip: bool) -> float:
            total = 0.0
            for (sx, sy), (cx, cy) in samples:
                dy = float(sy) - self.net_offset[1]
                got = (float(sx) - self.net_offset[0], -dy if flip else dy)
                total += (got[0] - cx) ** 2 + (got[1] - cy) ** 2
            return total

        self.y_flip = error(True) < error(False)
        return self.y_flip

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"CoordinateBridge(net_offset={self.net_offset}, "
                f"y_flip={self.y_flip})")


def sumo_angle_to_heading(angle_deg: float, *, y_flip: bool = False) -> float:
    """SUMO's compass angle (degrees, clockwise from north) as a mathematical
    heading in radians, counter-clockwise from +x, wrapped to ``[-pi, pi)``.
    ``y_flip`` mirrors the y axis for a frame whose y points the other way."""
    heading = math.radians(90.0 - float(angle_deg))
    if y_flip:
        heading = -heading
    return (heading + math.pi) % (2.0 * math.pi) - math.pi
