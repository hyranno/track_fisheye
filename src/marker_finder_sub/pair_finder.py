from tracking_info import TrackingInfo

import numpy


class PairFinder:
    def __init__(
        self,
        pairing_threshold: int = 80,
    ):
        self.pairing_threshold = pairing_threshold

    def find(
        self,
        points_position: numpy.ndarray,
        points_rotation: numpy.ndarray,
        pairing_values: numpy.ndarray
    ) -> list[TrackingInfo]:
        markers = []

        pairing_sorted_indices = numpy.argsort(pairing_values, axis=1)[::-1]
        for i in range(len(pairing_values)):
            axis0_index = pairing_sorted_indices[i][0]
            axis1_index = pairing_sorted_indices[i][1]
            axis0_val = pairing_values[i][axis0_index]
            axis1_val = pairing_values[i][axis1_index]
            if ((self.pairing_threshold < axis0_val) and (self.pairing_threshold < axis1_val)):
                position = points_position[i]
                axis0 = points_rotation[axis0_index]
                axis1 = points_rotation[axis1_index]
                a0 = axis0 - position
                a1 = axis1 - position
                z = a0[0]*a1[1] - a0[1]*a1[0]  # z part of cross product
                axy = (a1, a0) if (z < 0) else (a0, a1)
                marker = TrackingInfo(position, axy[0], axy[1])
                markers.append(marker)
        return markers
