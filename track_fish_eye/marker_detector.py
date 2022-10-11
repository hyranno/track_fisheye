import wgpu_util
import texture_util
import cv_util
from quick_track_marker import QuickTrackMarker
from smooth_threshold_filter import SmoothThresholdFilter
from fisheye_detector import FisheyeDetector
from marker_pair_finder import MarkerPairFinder

import sys
import cv2
import numpy
import wgpu
from wgpu.gui.auto import run


class MarkerDetector:
    def __init__(self, device: wgpu.GPUDevice, src_view: wgpu.GPUTextureView):
        self.device = device
        texture_size = src_view.texture.size
        self.preproced_view = texture_util.create_buffer_texture(device, texture_size).create_view()
        self.matched_view = texture_util.create_buffer_texture(device, texture_size).create_view()

        self.preprocess_filter = SmoothThresholdFilter(
            device, src_view, self.preproced_view, self.preproced_view.texture.format
        )
        self.pattern_matcher = FisheyeDetector(
            device, self.preproced_view, self.matched_view, self.matched_view.texture.format
        )
        self.pair_finder = MarkerPairFinder(device, self.preproced_view, self.matched_view)

        self.match_command_buffers = []
        self.match_command_buffers.extend(self.preprocess_filter.command_buffers)
        self.match_command_buffers.extend(self.pattern_matcher.command_buffers)

    def detect(self) -> list[QuickTrackMarker]:
        self.device.queue.submit(self.match_command_buffers)
        markers = self.pair_finder.find()
        return markers


if __name__ == "__main__":
    args = sys.argv
    # init image path from args or default
    path = "resources/image.png"
    print("processing " + path)
    device, context = wgpu_util.create_sized_canvas((512, 512), "image io example")
    context_texture_view = context.get_current_texture()
    context_texture_format = context.get_preferred_format(device.adapter)

    src = cv2.imread(path, -1)
    src_view = cv_util.cvimage_to_texture(src, device).create_view()
    detector = MarkerDetector(device, src_view)

    markers = detector.detect()
    for m in markers:
        cv_util.draw_wireframe_cube(src, m.size, m.points2d[0], m.quat)
    texture_preview_marker = cv_util.cvimage_to_texture(src, device)
    texture_util.draw_texture_on_texture(texture_preview_marker, context_texture_view, context_texture_format, device)
    run()
