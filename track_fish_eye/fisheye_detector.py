import wgpu_util
import texture_util
import cv_util

from multiscale_matcher1d import MultiScaleMatcher1d
from shader.match_result_integrate import MatchResultIntegrateShader

import sys
import numpy
import wgpu
from wgpu.gui.auto import run


def fisheye_kernel_ndarray() -> numpy.ndarray:
    kernel = numpy.array([
        1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0
    ], dtype=numpy.dtype('<f'), order='C')
    return kernel


def fisheye_PN_kernel_ndarray(kernel_half: numpy.ndarray) -> numpy.ndarray:
    kernel = numpy.concatenate((kernel_half, -kernel_half), axis=None)
    return kernel


class FisheyeDetector:
    def __init__(
        self,
        device: wgpu.GPUDevice,
        src_view: wgpu.GPUTextureView,
        dest_view: wgpu.GPUTextureView,
        dest_format: wgpu.TextureFormat
    ):
        self.device = device
        self.command_buffers = []
        self.texture_size = src_view.texture.size
        linear_sampler = device.create_sampler(min_filter="linear", mag_filter="linear")
        nearest_sampler = device.create_sampler(min_filter="nearest", mag_filter="nearest")

        kernel = fisheye_kernel_ndarray()
        scale_min = 0.05
        scale_max = 0.2
        scale_step = 1.1
        match_threshold = 0.6

        self.match1d_view_vertical = texture_util.create_buffer_texture(
            device, self.texture_size, wgpu.TextureFormat.rgba8snorm,
        ).create_view()
        self.match1d_view_horizontal = texture_util.create_buffer_texture(
            device, self.texture_size, wgpu.TextureFormat.rgba8snorm,
        ).create_view()
        self.match1d_vertical = MultiScaleMatcher1d(
            device,
            src_view,
            self.match1d_view_vertical,
            self.match1d_view_vertical.texture.format,
            kernel,
            numpy.array([0.0, 1.0]),
            scale_min,
            scale_max,
            scale_step,
        )
        self.match1d_horizontal = MultiScaleMatcher1d(
            device,
            src_view,
            self.match1d_view_horizontal,
            self.match1d_view_horizontal.texture.format,
            kernel,
            numpy.array([1.0, 0.0]),
            scale_min,
            scale_max,
            scale_step,
        )
        self.command_buffers.extend(self.match1d_vertical.command_buffers)
        self.command_buffers.extend(self.match1d_horizontal.command_buffers)

        self.match_result_shader = MatchResultIntegrateShader(
            device, self.match1d_view_vertical.texture.format
        )
        self.match_result_pipeline = self.match_result_shader.create_render_pipeline([
            {
                "format": dest_format,
                "blend": texture_util.BLEND_STATE_REPLACE,
            },
        ])
        self.match_result_bind = self.match_result_shader.create_bind_group(
            self.match1d_view_vertical,
            self.match1d_view_horizontal,
            nearest_sampler,
            match_threshold,
        )
        command_encoder = self.device.create_command_encoder()
        wgpu_util.push_2drender_pass(
            command_encoder, dest_view, self.match_result_pipeline, self.match_result_bind
        )
        self.command_buffers.append(command_encoder.finish())

        return

    def draw(self) -> None:
        self.device.queue.submit(self.command_buffers)


if __name__ == "__main__":
    args = sys.argv
    # init image path from args or default
    src_path = "resources/image_preproced.png"
    dest_path = "resources/image_matched.png"
    print("processing " + src_path)
    device, context = wgpu_util.create_sized_canvas((512, 512), "image io example")
    context_texture_view = context.get_current_texture()
    context_texture_format = context.get_preferred_format(device.adapter)
    texture_src = cv_util.imread_texture(src_path, device)
    dest_view = texture_util.create_buffer_texture(device, texture_src.size).create_view()
    filter = FisheyeDetector(device, texture_src.create_view(), dest_view, dest_view.texture.format)

    filter.draw(dest_view)
    cv_util.imwrite_texture(dest_view.texture, 4, dest_path, device)
    texture_util.draw_texture_on_texture(dest_view.texture, context_texture_view, context_texture_format, device)
    run()