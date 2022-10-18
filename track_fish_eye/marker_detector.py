import wgpu_util
import texture_util
import cv_util
from quick_track_marker import QuickTrackMarker
from smooth_threshold_filter import SmoothThresholdFilter
from fisheye_detector import FisheyeDetector
from marker_pair_finder import MarkerPairFinder

import sys
import cv2
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
            device, self.preproced_view, self.matched_view, self.matched_view.texture.format,
            0.05, 0.4, 1.1,
        )
        self.pair_finder = MarkerPairFinder(device, self.preproced_view, self.matched_view)

    def detect(self) -> list[QuickTrackMarker]:
        command_encoder = self.device.create_command_encoder()
        self.preprocess_filter.push_passes_to(command_encoder)
        self.pattern_matcher.push_passes_to(command_encoder)
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
    texture_util.draw_texture_on_texture(texture_preview_marker, context_texture_view, context_texture_format, device)
    run()
