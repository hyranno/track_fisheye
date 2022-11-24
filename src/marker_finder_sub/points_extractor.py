import cv_util
from xmeans import XMeans

import numpy
import wgpu


class PointsExtractor:
    def __init__(
        self,
        device: wgpu.GPUDevice,
        src_view: wgpu.GPUTextureView,
    ):
        self.device = device
        self.texture_size = src_view.texture.size
        self.src_view = src_view

    def extract(self) -> tuple[numpy.ndarray, numpy.ndarray]:
        src_np = cv_util.texture_to_cvimage(self.device, self.src_view.texture, 4)

        positives = list(zip(*numpy.where(127 < src_np[:, :, 2])))
        negatives = list(zip(*numpy.where(127 < src_np[:, :, 1])))
        # """
        max_depth = 8  # 2^8 = 256 data points
        min_distance = 3.0
        points_position = numpy.array([[positives[0][1], positives[0][0]]])
        if 2 <= len(positives):
            xm_positives = XMeans(self.device, numpy.array(positives), max_depth, min_distance)
            clusters = xm_positives.process().read_clusters_valid_sized()
            points_position = numpy.array([[c.mean[1], c.mean[0]] for c in clusters])
        points_rotation = numpy.array([[negatives[0][1], negatives[0][0]]])
        if 2 <= len(negatives):
            xm_negatives = XMeans(self.device, numpy.array(negatives), max_depth, min_distance)
            clusters = xm_negatives.process().read_clusters_valid_sized()
            points_rotation = numpy.array([[c.mean[1], c.mean[0]] for c in clusters])

        return (points_position, points_rotation)
