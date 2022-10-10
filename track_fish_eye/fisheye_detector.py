import wgpu_util
import texture_util
import cv_util

import shader.match_multi_scale
import shader.match_result_integrate

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
    def __init__(self, device: wgpu.GPUDevice, src_view: wgpu.GPUTextureView, dest_format: wgpu.TextureFormat):
        self.device = device
        self.texture_size = src_view.texture.size
        linear_sampler = device.create_sampler(min_filter="linear", mag_filter="linear")
        nearest_sampler = device.create_sampler(min_filter="nearest", mag_filter="nearest")

        self.match_threshold = 0.6
        self.match_min = 0.05
        self.scale_max = 4.0
        self.scale_step = 1.1
        self.kernel = fisheye_kernel_ndarray()
        self.match1d_view_vertical = texture_util.create_buffer_texture(device, self.texture_size).create_view()
        self.match1d_view_horizontal = texture_util.create_buffer_texture(device, self.texture_size).create_view()
        self.match1d_shader = shader.match_multi_scale.MatchMultiScale1dShader(device, src_view.texture.format)
        self.match1d_pipeline = self.match1d_shader.create_render_pipeline([
            {
                "format": self.match1d_view_vertical.texture.format,
                "blend": texture_util.BLEND_STATE_REPLACE,
            },
        ])
        self.match1d_bind_vertical = self.match1d_shader.create_bind_group(
            src_view,
            linear_sampler,
            self.kernel,
            self.match_threshold,
            (0.0, self.match_min),
            self.scale_max,
            self.scale_step,
        )
        self.match1d_bind_horizontal = self.match1d_shader.create_bind_group(
            src_view,
            linear_sampler,
            self.kernel,
            self.match_threshold,
            (self.match_min, 0.0),
            self.scale_max,
            self.scale_step,
        )

        self.match_result_shader = shader.match_result_integrate.MatchResultIntegrateShader(
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
            nearest_sampler,
            self.match1d_view_horizontal,
            nearest_sampler,
        )

        return

    def draw(self, dest_view: wgpu.GPUTextureView) -> None:
        command_encoder = self.device.create_command_encoder()

        wgpu_util.push_2drender_pass(
            command_encoder, self.match1d_view_horizontal, self.match1d_pipeline, self.match1d_bind_horizontal
        )
        wgpu_util.push_2drender_pass(
            command_encoder, self.match1d_view_vertical, self.match1d_pipeline, self.match1d_bind_vertical
        )
        wgpu_util.push_2drender_pass(
            command_encoder, dest_view, self.match_result_pipeline, self.match_result_bind
        )

        self.device.queue.submit([command_encoder.finish()])


if __name__ == "__main__":
    args = sys.argv
    # init image path from args or default
    src_path = "resources/image_preproced.png"
    print("processing " + src_path)
    device, context = wgpu_util.create_sized_canvas((512, 512), "image io example")
    context_texture_view = context.get_current_texture()
    context_texture_format = context.get_preferred_format(device.adapter)
    texture_src = cv_util.imread_texture(src_path, device)
    dest_view = texture_util.create_buffer_texture(device, texture_src.size).create_view()
    filter = FisheyeDetector(device, texture_src.create_view(), dest_view.texture.format)

    filter.draw(dest_view)
    texture_util.draw_texture_on_texture(dest_view.texture, context_texture_view, context_texture_format, device)
    run()
