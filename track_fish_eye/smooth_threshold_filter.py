import wgpu_util
import texture_util
import cv_util
import util

import shader.grayscale
from shader.filter1d import Filter1dShader
import shader.smooth_threshold


import sys
import array
import math
import numpy
import wgpu
from wgpu.gui.auto import run


def gaussian_kernel_ndarray(size: int) -> numpy.ndarray:
    return numpy.array(util.gaussian_kernel(size), dtype=numpy.float32)


class SmoothThresholdFilter:
    def __init__(
        self,
        device: wgpu.GPUDevice,
        src_view: wgpu.GPUTextureView,
        dest_view: wgpu.GPUTextureView,
        dest_format: wgpu.TextureFormat,
        gaussian_kernel_size: int = 61,
        edge: float = 0.1,
    ):
        self.device = device
        self.dest_view = dest_view
        self.texture_size = src_view.texture.size
        linear_sampler = device.create_sampler(min_filter="linear", mag_filter="linear")

        self.grayscale_view = texture_util.create_buffer_texture(device, self.texture_size).create_view()
        self.grayscale_shader = shader.grayscale.GrayscaleShader(device, src_view.texture.format)
        self.grayscale_pipeline = self.grayscale_shader.create_render_pipeline([
            {
                "format": self.grayscale_view.texture.format,  # context_texture_format,
                "blend": texture_util.BLEND_STATE_REPLACE,
            },
        ])
        self.grayscale_bind = self.grayscale_shader.create_bind_group(
            src_view,
            linear_sampler
        )

        self.gaussian_view = texture_util.create_buffer_texture(device, self.texture_size).create_view()
        self.gaussian_tmp_view = texture_util.create_buffer_texture(device, self.texture_size).create_view()
        gaussian_kernel = numpy.array(util.gaussian_kernel(gaussian_kernel_size), dtype=numpy.float32)
        self.gaussian_shader = Filter1dShader(
            device,
            self.grayscale_view.texture.format,
        )
        self.gaussian_pipeline = self.gaussian_shader.create_render_pipeline([
            {
                "format": self.grayscale_view.texture.format,
                "blend": texture_util.BLEND_STATE_REPLACE,
            },
        ])
        self.gaussian_bind_horizontal = self.gaussian_shader.create_bind_group(
            self.grayscale_view,
            linear_sampler,
            gaussian_kernel,
            (1, 0),
        )
        self.gaussian_bind_vertical = self.gaussian_shader.create_bind_group(
            self.gaussian_tmp_view,
            linear_sampler,
            gaussian_kernel,
            (0, 1),
        )

        self.smooth_threshold_view = texture_util.create_buffer_texture(device, self.texture_size).create_view()
        self.smooth_threshold_shader = shader.smooth_threshold.SmoothThresholdShader(
            device,
            self.gaussian_view.texture.format
        )
        self.smooth_threshold_pipeline = self.smooth_threshold_shader.create_render_pipeline([
            {
                "format": dest_format,
                "blend": texture_util.BLEND_STATE_REPLACE,
            },
        ])
        self.smooth_threshold_bind = self.smooth_threshold_shader.create_bind_group(
            linear_sampler,
            self.grayscale_view,
            self.gaussian_view,
            (-edge, edge)
        )

        return

    def push_passes_to(self, encoder: wgpu.GPUCommandEncoder):
        wgpu_util.push_2drender_pass(
            encoder, self.grayscale_view, self.grayscale_pipeline, self.grayscale_bind
        )
        wgpu_util.push_2drender_pass(
            encoder, self.gaussian_tmp_view, self.gaussian_pipeline, self.gaussian_bind_horizontal
        )
        wgpu_util.push_2drender_pass(
            encoder, self.gaussian_view, self.gaussian_pipeline, self.gaussian_bind_vertical
        )
        wgpu_util.push_2drender_pass(
            encoder, self.dest_view, self.smooth_threshold_pipeline, self.smooth_threshold_bind
        )

    def draw(self) -> None:
        command_encoder = self.device.create_command_encoder()
        self.push_passes_to(command_encoder)
        self.device.queue.submit([command_encoder.finish()])


if __name__ == "__main__":
    args = sys.argv
    # init image path from args or default
    src_path = "resources/image.png"
    dest_path = "resources/image_preproced.png"
    print("processing " + src_path)
    device, context = wgpu_util.create_sized_canvas((512, 512), "image io example")
    context_texture_view = context.get_current_texture()
    context_texture_format = context.get_preferred_format(device.adapter)
    texture_src = cv_util.imread_texture(src_path, device)
    dest_view = texture_util.create_buffer_texture(device, texture_src.size).create_view()
    filter = SmoothThresholdFilter(
        device, texture_src.create_view(), dest_view, dest_view.texture.format
    )

    filter.draw()
    cv_util.imwrite_texture(dest_view.texture, 4, dest_path, device)
    texture_util.draw_texture_on_texture(dest_view.texture, context_texture_view, context_texture_format, device)
    run()
