import wgpu_util
import texture_util
import cv_util
from quick_track_marker import QuickTrackMarker

from smooth_threshold_filter_cs import SmoothThresholdFilter
from fisheye_detector_cs import FisheyeDetector
from marker_pair_finder_cs import MarkerPairFinder

import sys
import cv2
import numpy
import wgpu
from wgpu.gui.auto import run


def fisheye_kernel_ndarray() -> numpy.ndarray:
    kernel = numpy.array([
        1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0
    ], dtype=numpy.dtype('<f'), order='C')
    return kernel


class MarkerDetector:
    def __init__(self, device: wgpu.GPUDevice, src_view: wgpu.GPUTextureView):
        self.device = device
        texture_size = src_view.texture.size
        self.preproced_view = texture_util.create_buffer_texture(device, texture_size).create_view()
        self.matched_view = texture_util.create_buffer_texture(device, texture_size).create_view()
        kernel = fisheye_kernel_ndarray()

        self.preprocess_filter = SmoothThresholdFilter(device, src_view, self.preproced_view)
        self.pattern_matcher = FisheyeDetector(
            device, self.preproced_view, self.matched_view,
            20, 200, 1.1, 0.6, kernel
        )
        self.pair_finder = MarkerPairFinder(device, self.preproced_view, self.matched_view, kernel)

    def detect(self) -> list[QuickTrackMarker]:
        command_encoder = self.device.create_command_encoder()
        self.preprocess_filter.push_compute_passes_to(command_encoder)
        self.pattern_matcher.push_compute_passes_to(command_encoder)
        self.device.queue.submit([command_encoder.finish()])
        markers = self.pair_finder.find()
        return markers


if __name__ == "__main__":
    args = sys.argv
    # init image path from args or default
    path = "resources/image.png"
    print("processing " + path)
    device, context = wgpu_util.create_sized_canvas((512, 512), "marker detector example")
    context_texture_view = context.get_current_texture()
    context_texture_format = context.get_preferred_format(device.adapter)

    src = cv2.imread(path, -1)
    src_view = cv_util.cvimage_to_texture(src, device).create_view()
    detector = MarkerDetector(device, src_view)

    markers = detector.detect()
    for m in markers:
        cv_util.draw_tracked_marker(src, m)
    texture_preview_marker = cv_util.cvimage_to_texture(src, device)
    texture_util.draw_texture_on_texture(
        # detector.preproced_view.texture, context_texture_view, context_texture_format, device
        # detector.matched_view.texture, context_texture_view, context_texture_format, device
        texture_preview_marker, context_texture_view, context_texture_format, device
    )
    run()
