
import numpy
import quaternion


class QuickTrackMarker:
    def __init__(
        self,
        position: tuple[float, float],
        axis: tuple[tuple[float, float], tuple[float, float]],
    ):
        self.points2d = (position, axis)
        a0 = (axis[0][0] - position[0], axis[0][1] - position[1])
        a1 = (axis[1][0] - position[0], axis[1][1] - position[1])
        if (a0[0]*a1[1] - a0[1]*a1[0] < 0):  # z part of cross product
            tmp = a0
            a0 = a1
            a1 = tmp
        Ca = 1
        Cb = -(a0[0]*a0[0] + a0[1]*a0[1] + a1[0]*a1[0] + a1[1]*a1[1])
        Cc = (a0[0]*a1[1] - a0[1]*a1[0]) * (a0[0]*a1[1] - a0[1]*a1[0])
        c = numpy.sqrt((-Cb + numpy.sqrt(Cb*Cb - 4*Ca*Cc)) / 2*Ca)
        v0 = numpy.array([a0[0], a0[1], numpy.sqrt(c*c - a0[0]*a0[0] - a0[1]*a0[1])]) / c
        v1 = numpy.array([a1[0], a1[1], numpy.sqrt(c*c - a1[0]*a1[0] - a1[1]*a1[1])]) / c
        v2 = numpy.cross(v0, v1)

        self.size = c
        dcm = numpy.array([
            [v0[0], v1[0], v2[0]],
            [v0[1], v1[1], v2[1]],
            [v0[2], v1[2], v2[2]],
        ])
        self.quat = quaternion.from_rotation_matrix(dcm)

    def __repr__(self) -> str:
        return str([
            "points 2d: " + str(self.points2d),
            "size: " + str(self.size),
            "quaternion: " + str(self.quat),
        ])
