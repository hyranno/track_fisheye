import cv_util

import numpy
import wgpu
from pyclustering.cluster import xmeans


def xmcluster(points) -> xmeans.xmeans:
    initial_center = xmeans.kmeans_plusplus_initializer(points, 1).initialize()
    xm = xmeans.xmeans(points, initial_center, ccore=True)
    xm.process()
    return xm


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

        points_position = numpy.array([])
        if 0 < len(positives):
            xm_positives = xmcluster(positives)
            points_position = numpy.fliplr(numpy.array(xm_positives.get_centers()))
        points_rotation = numpy.array([])
        if 0 < len(negatives):
            xm_negatives = xmcluster(negatives)
            points_rotation = numpy.fliplr(numpy.array(xm_negatives.get_centers()))

        return (points_position, points_rotation)
