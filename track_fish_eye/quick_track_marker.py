
import numpy


class QuickTrackMarker:
    def __init__(
        self,
        position: tuple[float, float],
        axis0: tuple[float, float],
        axis1: tuple[float, float],
    ):
        self.position = position
        self.axis0 = axis0
        self.axis1 = axis1

    def __repr__(self) -> str:
        return str([
            "position: " + str(self.position),
            "axis0: " + str(self.axis0),
            "axis1: " + str(self.axis1),
        ])
