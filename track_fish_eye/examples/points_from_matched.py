import mymodule
import wgpu_util
import texture_util
import cv_util

import shader.match_multi_scale
import shader.match_result_integrate

import array
import math
import numpy
import wgpu
from wgpu.gui.auto import run
from pyclustering.cluster import xmeans


def xmcluster(points) -> xmeans.xmeans:
    initial_center = xmeans.kmeans_plusplus_initializer(points, 1).initialize()
    xm = xmeans.xmeans(points, initial_center, ccore=True)
    xm.process()
    return xm


resolution = 512, 512
device, context = wgpu_util.create_sized_canvas(resolution, "matched points example")
context_texture_view = context.get_current_texture()
context_texture_format = context.get_preferred_format(device.adapter)

texture_src = cv_util.imread_texture("resources/image_matched.png", device)
view_src = texture_src.create_view()
shape = texture_src.size[1], texture_src.size[0], 4  # for numpy.ndarray

src_np = cv_util.texture_to_cvimage(texture_src, shape, device)

threshold = 50
positives = list(zip(*numpy.where(src_np[:, :, 0] > 127+threshold)))
negatives = list(zip(*numpy.where(src_np[:, :, 0] < 127-threshold)))

xm_positives = xmcluster(positives)
xm_negatives = xmcluster(negatives)

points_position = xm_positives.get_centers()
points_rotation = xm_negatives.get_centers()

print(points_position)
print(points_rotation)
