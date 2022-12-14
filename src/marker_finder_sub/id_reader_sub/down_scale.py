import wgpu_util
import texture_util
import util

from shaders.filter1d import Filter1dShader
from shaders.draw_texture import TextureShader

import numpy
import wgpu


class DownScaleFilter:
    def __init__(
        self,
        device: wgpu.GPUDevice,
        src_view: wgpu.GPUTextureView,
        dest_view: wgpu.GPUTextureView,
        dest_format: wgpu.TextureFormat,
    ):
        self.device = device
        self.dest_view = dest_view
        self.texture_size = src_view.texture.size
        linear_sampler = device.create_sampler(min_filter="linear", mag_filter="linear")
        ratio = float(dest_view.texture.size[0]) / float(src_view.texture.size[0])

        gaussian_kernel_size = 2 * int(ratio / 2.0) + 1
        self.gaussian_view = texture_util.create_buffer_texture(device, self.texture_size).create_view()
        self.gaussian_tmp_view = texture_util.create_buffer_texture(device, self.texture_size).create_view()
        gaussian_kernel = numpy.array(util.gaussian_kernel(gaussian_kernel_size), dtype=numpy.float32)
        self.gaussian_shader = Filter1dShader(
            device,
            src_view.texture.format,
        )
        self.gaussian_pipeline = self.gaussian_shader.create_render_pipeline([
            {
                "format": src_view.texture.format,
                "blend": texture_util.BLEND_STATE_REPLACE,
            },
        ])
        self.gaussian_bind_horizontal = self.gaussian_shader.create_bind_group(
            src_view,
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

        self.resample_shader = TextureShader(device, self.gaussian_view.texture.format)
        self.resample_pipeline = self.resample_shader.create_render_pipeline([
            {
                "format": dest_format,
                "blend": texture_util.BLEND_STATE_REPLACE,
            },
        ])
        self.resample_bind = self.resample_shader.create_bind_group(self.gaussian_view, linear_sampler)

    def push_passes_to(self, encoder: wgpu.GPUCommandEncoder):
        wgpu_util.push_2drender_pass(
            encoder, self.gaussian_tmp_view, self.gaussian_pipeline, self.gaussian_bind_horizontal
        )
        wgpu_util.push_2drender_pass(
            encoder, self.gaussian_view, self.gaussian_pipeline, self.gaussian_bind_vertical
        )
        wgpu_util.push_2drender_pass(
            encoder, self.dest_view, self.resample_pipeline, self.resample_bind
        )

    def draw(self) -> None:
        command_encoder = self.device.create_command_encoder()
        self.push_passes_to(command_encoder)
        self.device.queue.submit([command_encoder.finish()])
