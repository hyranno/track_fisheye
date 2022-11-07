
import numpy
# import quaternion


class TrackingInfo:
    def __init__(
        self,
        position: tuple[float, float],
        ax: tuple[float, float],
        ay: tuple[float, float]
    ):
        self.position = position
        self.ax = ax
        self.ay = ay

        Ca = 1
        Cb = -(ax[0]*ax[0] + ax[1]*ax[1] + ay[0]*ay[0] + ay[1]*ay[1])
        Cc = (ax[0]*ay[1] - ax[1]*ay[0]) * (ax[0]*ay[1] - ax[1]*ay[0])
        c = numpy.sqrt((-Cb + numpy.sqrt(Cb*Cb - 4*Ca*Cc)) / 2*Ca)
        v0 = numpy.array([ax[0], ax[1], numpy.sqrt(c*c - ax[0]*ax[0] - ax[1]*ax[1])]) / c
        v1 = numpy.array([ay[0], ay[1], numpy.sqrt(c*c - ay[0]*ay[0] - ay[1]*ay[1])]) / c
        v2 = numpy.cross(v0, v1)

        self.size = c
        dcm = numpy.array([
            [v0[0], v1[0], v2[0]],
            [v0[1], v1[1], v2[1]],
            [v0[2], v1[2], v2[2]],
        ])
        self.dcm = dcm  # for precision
        # self.quat = quaternion.from_rotation_matrix(dcm)
        self.id: int | None = None

    def __repr__(self) -> str:
        return str([
            "points 2d: " + str(self.points2d),
            "size: " + str(self.size),
            "quaternion: " + str(self.quat),
        ])
