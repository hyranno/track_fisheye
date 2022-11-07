import wgpu_util
import texture_util

from .match1d import Match1dShader
from .integrate_minmax import IntegrateMinMaxShader

import numpy
import wgpu


class MultiScaleMatcher1d:
    def __init__(
        self,
        device: wgpu.GPUDevice,
        src_view: wgpu.GPUTextureView,
        dest_view: wgpu.GPUTextureView,
        dest_format: wgpu.TextureFormat,
        kernel: numpy.ndarray,
        direction: numpy.ndarray,
        scale_min: int,
        scale_max: int,
        scale_step: float,
    ):
        self.device = device
        self.texture_size = src_view.texture.size
        linear_sampler = device.create_sampler(min_filter="linear", mag_filter="linear")

        self.ranges = []
        r = scale_min
        while (r <= scale_max):
            self.ranges.append(r)
            r *= scale_step

        self.match1d_shader = Match1dShader(device, src_view.texture.format)
        self.match1d_pipeline = self.match1d_shader.create_render_pipeline([
            {
                "format": wgpu.TextureFormat.rgba8snorm,
                "blend": texture_util.BLEND_STATE_REPLACE,
            },
        ])
        self.match1d_views = [
            texture_util.create_buffer_texture(device, self.texture_size, wgpu.TextureFormat.rgba8snorm).create_view()
            for i in range(len(self.ranges))
        ]
        self.match1d_binds = [
            self.match1d_shader.create_bind_group(
                src_view, linear_sampler, kernel, direction * r / self.texture_size[0:2]
            )
            for r in self.ranges
        ]

        self.integrate_shader = IntegrateMinMaxShader(device, src_view.texture.format)
        self.integrate_pipeline = self.integrate_shader.create_render_pipeline([
            {
                "format": wgpu.TextureFormat.rgba8snorm,
                "blend": texture_util.BLEND_STATE_REPLACE,
            },
        ])
        self.integrate_views = [
            texture_util.create_buffer_texture(device, self.texture_size, wgpu.TextureFormat.rgba8snorm).create_view()
            for i in range(len(self.ranges)-1)
        ]
        self.integrate_views[0] = dest_view
        self.integrate_views.append(self.match1d_views[len(self.ranges)-1])
        self.integrate_binds = [
            self.integrate_shader.create_bind_group(
                self.integrate_views[i+1],
                self.match1d_views[i],
                linear_sampler,
            )
            for i in range(len(self.ranges)-1)
        ]

    def push_passes_to(self, encoder: wgpu.GPUCommandEncoder):
        for i in range(len(self.match1d_binds)):
            wgpu_util.push_2drender_pass(
                encoder, self.match1d_views[i], self.match1d_pipeline, self.match1d_binds[i]
            )
        for i in reversed(range(len(self.integrate_binds))):
            wgpu_util.push_2drender_pass(
                encoder, self.integrate_views[i], self.integrate_pipeline, self.integrate_binds[i]
            )

    def draw(self) -> None:
        command_encoder = self.device.create_command_encoder()
        self.push_passes_to(command_encoder)
        self.device.queue.submit([command_encoder.finish()])
