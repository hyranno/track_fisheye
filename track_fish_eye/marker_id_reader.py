import wgpu_util
import texture_util
import cv_util
from marker_format import MarkerFormat
from shader.img_to_marker import ImgToMarkerShader

import sys
import cv2
import numpy
import wgpu
from wgpu.gui.auto import run

from marker_detector import MarkerDetector, DetectorParams


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
        self.sampler = device.create_sampler(min_filter="linear", mag_filter="linear")
        marker_size = (marker_format.length, marker_format.length, 1)

        self.marker_view = texture_util.create_buffer_texture(device, marker_size).create_view()
        self.i2m_shader = ImgToMarkerShader(device, src_view.texture.format)
        self.i2m_pipeline = self.i2m_shader.create_render_pipeline([
            {
                "format": self.marker_view.texture.format,
                "blend": texture_util.BLEND_STATE_REPLACE,
            },
        ])

    def read(
        self,
        pos: tuple[float, float],
        axis0: tuple[float, float],
        axis1: tuple[float, float]
    ) -> int:
        i2m_bind = self.i2m_shader.create_bind_group(
            self.src_view, self.sampler,
            pos, axis0, axis1,
            self.marker_format.length
        )
        command_encoder = device.create_command_encoder()
        wgpu_util.push_2drender_pass(
            command_encoder, self.marker_view, self.i2m_pipeline, i2m_bind
        )
        device.queue.submit([command_encoder.finish()])
        marker = (cv2.cvtColor(
            cv_util.texture_to_cvimage(self.device, self.marker_view.texture, 4),
            cv2.COLOR_BGRA2GRAY
        ) / 255).astype(numpy.uint8)
        return self.marker_format.extract_id(marker)


if __name__ == "__main__":
    args = sys.argv
    # init image path from args or default
    path = "resources/image.png"
    path_preproced = "resources/image_preproced.png"
    print("processing " + path)
    marker_format = MarkerFormat([1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0])

    device, context = wgpu_util.create_sized_canvas((512, 512), "marker detector example")
    context_texture_view = context.get_current_texture()
    context_texture_format = context.get_preferred_format(device.adapter)

    src = cv2.imread(path, -1)
    src_view = cv_util.cvimage_to_texture(src, device).create_view()
    detector = MarkerDetector(device, src_view, marker_format, DetectorParams(20, 200, 1.08, 0.6))

    preproced = cv2.imread(path_preproced, -1)
    preproced_view = cv_util.cvimage_to_texture(preproced, device).create_view()
    id_reader = MarkerIdReader(device, preproced_view, marker_format)

    markers = detector.detect()
    for m in markers:
        cv_util.draw_tracked_marker(src, m)
        id = id_reader.read(m.points2d[0], m.points2d[1][0], m.points2d[1][1])
        # draw id
        print(id)
    texture_preview_marker = cv_util.cvimage_to_texture(src, device)
    texture_util.draw_texture_on_texture(texture_preview_marker, context_texture_view, context_texture_format, device)
    run()
