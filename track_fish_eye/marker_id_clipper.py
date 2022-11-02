import wgpu_util
import texture_util
import cv_util
import util

from shader.grayscale import GrayscaleShader
from shader.affine import AffineShader
from down_scale import DownScaleFilter

from marker_detector import MarkerDetector, DetectorParams, MarkerFormat

import sys
import numpy
import wgpu
from wgpu.gui.auto import run

# grayscale
# affine
# down_scale
# threshold (cpu?)


class MarkerIdClipper:
    def __init__(
        self,
        device: wgpu.GPUDevice,
        src_view: wgpu.GPUTextureView,
        marker_id_resolution: tuple[int, int],
    ):
        self.device = device
        self.src_view = src_view
        self.texture_size = src_view.texture.size
        self.resolution = marker_id_resolution
        linear_sampler = device.create_sampler(min_filter="linear", mag_filter="linear")
        self.linear_sampler = linear_sampler

        self.grayscale_view = texture_util.create_buffer_texture(device, self.texture_size).create_view()
        self.grayscale_shader = GrayscaleShader(device, src_view.texture.format)
        self.grayscale_pipeline = self.grayscale_shader.create_render_pipeline([
            {
                "format": self.grayscale_view.texture.format,
                "blend": texture_util.BLEND_STATE_REPLACE,
            },
        ])
        self.grayscale_bind = self.grayscale_shader.create_bind_group(
            src_view,
            linear_sampler
        )

        self.affine_view = texture_util.create_buffer_texture(device, self.texture_size).create_view()
        self.affine_shader = AffineShader(device, src_view.texture.format)
        self.affine_pipeline = self.affine_shader.create_render_pipeline([
            {
                "format": self.affine_view.texture.format,
                "blend": texture_util.BLEND_STATE_REPLACE,
            },
        ])

        self.down_scale_view = texture_util.create_buffer_texture(device, (*marker_id_resolution, 1)).create_view()
        self.down_scale_filter = DownScaleFilter(
            device, self.affine_view, self.down_scale_view, self.down_scale_view.texture.format
        )

    def clip(
        self,
        p0: tuple[float, float],
        pa0: tuple[float, float],
        pa1: tuple[float, float]
    ) -> numpy.ndarray:
        affine_bind = self.affine_shader.create_bind_group(
            self.grayscale_view,
            self.linear_sampler,
            self.calc_affine(p0, pa0, pa1)
        )
        encoder = self.device.create_command_encoder()
        wgpu_util.push_2drender_pass(
            encoder, self.grayscale_view, self.grayscale_pipeline, self.grayscale_bind
        )
        wgpu_util.push_2drender_pass(
            encoder, self.affine_view, self.affine_pipeline, affine_bind
        )
        self.down_scale_filter.push_passes_to(encoder)
        self.device.queue.submit([encoder.finish()])
        return self.threshold(cv_util.texture_to_cvimage(
            self.device,
            self.down_scale_view.texture, 4
        ))

    def calc_affine(
        self,
        p0: tuple[float, float],
        pa0: tuple[float, float],
        pa1: tuple[float, float]
    ) -> numpy.ndarray:
        p = numpy.array(p0)
        a0 = pa0 - p
        a1 = pa1 - p
        d = self.src_view.texture.size[0:2]
        Am = numpy.matrix([
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 0.5],
            [0.0, 0.0, 1.0],
        ])  # id area of marker
        Af = numpy.matrix([
            [a0[0], a1[0], 0.0],
            [a0[1], a1[1], 0.0],
            [0.0, 0.0, 1.0],
        ])  # rotation, skew
        Ap = numpy.matrix([
            [1.0, 0.0, p[0]],
            [0.0, 1.0, p[1]],
            [0.0, 0.0, 1.0],
        ])  # translation
        Auv = numpy.matrix([
            [1.0/d[0], 0.0, 0.0],
            [0.0, 1.0/d[1], 0.0],
            [0.0, 0.0, 1.0],
        ])  # px to uv
        return Auv @ Ap @ Af @ Am

    def threshold(self, image: numpy.ndarray) -> numpy.ndarray:
        img = image[:, :, 0]
        th = img.sum() / (img.shape[0] * img.shape[1])
        return numpy.vectorize(lambda v: 0 if v < th else 1)(img)


if __name__ == "__main__":
    args = sys.argv
    # init image path from args or default
    path = "resources/image.png"
    print("processing " + path)
    id_resolution = (128, 128)  # 4

    device, context = wgpu_util.create_sized_canvas(id_resolution, "marker detector example")
    context_texture_view = context.get_current_texture()
    context_texture_format = context.get_preferred_format(device.adapter)
    src_view = cv_util.imread_texture(path, device).create_view()

    detector = MarkerDetector(device, src_view, MarkerFormat(), DetectorParams(20, 200, 1.08, 0.6))
    markers = detector.detect()

    points = markers[0].points2d
    clipper = MarkerIdClipper(device, src_view, id_resolution)
    clipped = clipper.clip(points[0], points[1][0], points[1][1])

    print(points)
    print(clipped)
    texture_util.draw_texture_on_texture(
        clipper.down_scale_view.texture, context_texture_view, context_texture_format, device
    )
    run()
