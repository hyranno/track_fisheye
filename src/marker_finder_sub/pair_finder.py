from tracking_info import TrackingInfo

import numpy


class PairFinder:
    def __init__(
        self,
        pairing_threshold: int = 80,
    ):
        self.pairing_threshold = pairing_threshold

    def find_all(
        self,
        points_position: numpy.ndarray,
        points_rotation: numpy.ndarray,
        all_pairing_values: numpy.ndarray
    ) -> list[TrackingInfo]:
        markers_option = [
            self.find(points_position[i], points_rotation, all_pairing_values[i])
            for i in range(len(all_pairing_values))
        ]
        return [v for v in markers_option if not(v is None)]

    def find(
        self,
        position: numpy.ndarray,
        points_rotation: numpy.ndarray,
        pairing_values: numpy.ndarray
    ) -> TrackingInfo | None:
        if (len(points_rotation) < 2):
            return None
        pairing_sorted_indices = numpy.argsort(pairing_values)[::-1]
        axis0_index = pairing_sorted_indices[0]
        axis1_index = pairing_sorted_indices[1]
        axis1_val = pairing_values[axis1_index]
        if (axis1_val < self.pairing_threshold):
            return None
        axis0 = points_rotation[axis0_index]
        axis1 = points_rotation[axis1_index]
        a0 = axis0 - position
        a1 = axis1 - position
        z = a0[0]*a1[1] - a0[1]*a1[0]  # z part of cross product
        axy = (a1, a0) if (z < 0) else (a0, a1)
        return TrackingInfo(position, axy[0], axy[1])
