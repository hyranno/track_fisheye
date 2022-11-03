import wgpu_util
import texture_util
import cv_util
from marker_format import MarkerFormat

import sys
import cv2
import numpy
import wgpu
from wgpu.gui.auto import run

from marker_detector import MarkerDetector, DetectorParams
from marker_id_clipper import MarkerIdClipper


class MarkerIdReader:
    def __init__(
        self,
        device: wgpu.GPUDevice,
        src_view: wgpu.GPUTextureView,
        marker_format: MarkerFormat,
    ):
        self.device = device
        self.src_view = src_view
        self.marker_format = marker_format
        self.clipper = MarkerIdClipper(device, src_view, marker_format.id_resolution)
        self.sampler = device.create_sampler(min_filter="linear", mag_filter="linear")

    def read(
        self,
        pos: tuple[float, float],
        axis0: tuple[float, float],
        axis1: tuple[float, float]
    ) -> int:
        id_patch = self.clipper.clip(pos, axis0, axis1)
        return self.marker_format.extract_id(id_patch)


if __name__ == "__main__":
    args = sys.argv
    # init image path from args or default
    path = "resources/image.png"
    print("processing " + path)
    marker_format = MarkerFormat()

    device, context = wgpu_util.create_sized_canvas((512, 512), "marker id reader example")
    context_texture_view = context.get_current_texture()
    context_texture_format = context.get_preferred_format(device.adapter)

    src = cv2.imread(path, -1)
    src_view = cv_util.cvimage_to_texture(src, device).create_view()

    detector = MarkerDetector(device, src_view, marker_format, DetectorParams(20, 200, 1.08, 0.6))
    id_reader = MarkerIdReader(device, src_view, marker_format)

    markers = detector.detect()
    for m in markers:
        id = id_reader.read(m.points2d[0], m.points2d[1][0], m.points2d[1][1])
        cv2.putText(
            src, str(id), numpy.array(m.points2d[0]).astype(numpy.int32),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0, 255), 2
        )
        cv_util.draw_tracked_marker(src, m)
    texture_preview_marker = cv_util.cvimage_to_texture(src, device)
    texture_util.draw_texture_on_texture(texture_preview_marker, context_texture_view, context_texture_format, device)
    run()
