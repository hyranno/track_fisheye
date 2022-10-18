import wgpu_util
import texture_util

from shader.match1d_multiscale import Match1dMultiscaleShader
from shader.minmaxmean import MinMaxMeanShader

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
        scale_min: float,
        scale_max: float,
        scale_step: float,
    ):
        self.device = device
        self.texture_size = src_view.texture.size
        linear_sampler = device.create_sampler(min_filter="linear", mag_filter="linear")
        nearest_sampler = device.create_sampler(min_filter="nearest", mag_filter="nearest")

        self.scales = []
        s = scale_min
        while (s <= scale_max):
            self.scales.append(int(s * 400))  # TODO: scale_min, scale_max should be px
            s *= scale_step

        self.match1d_shader = Match1dMultiscaleShader(device, src_view.texture.format)
        self.match1d_pipeline = self.match1d_shader.create_render_pipeline([
            {
                "format": wgpu.TextureFormat.rgba8snorm,
                "blend": texture_util.BLEND_STATE_REPLACE,
            },
        ])
        self.match1d_view = device.create_texture(
            size=(self.texture_size[0], self.texture_size[1], len(self.scales)),
            usage=wgpu.TextureUsage.COPY_DST
            | wgpu.TextureUsage.COPY_SRC
            | wgpu.TextureUsage.TEXTURE_BINDING
            | wgpu.TextureUsage.RENDER_ATTACHMENT,
            dimension=wgpu.TextureDimension.d3,
            format=wgpu.TextureFormat.rgba8snorm,
            mip_level_count=1,
            sample_count=1,
        ).create_view()
        self.match1d_bind = self.match1d_shader.create_bind_group(
            src_view, linear_sampler, kernel, direction, numpy.array(self.scales)
        )

        self.integrate_shader = MinMaxMeanShader(device, src_view.texture.format)
        self.integrate_pipeline = self.integrate_shader.create_render_pipeline([
            {
                "format": wgpu.TextureFormat.rgba8snorm,
                "blend": texture_util.BLEND_STATE_REPLACE,
            },
        ])
        self.integrate_view = texture_util.create_buffer_texture(
            device, self.texture_size, wgpu.TextureFormat.rgba8snorm
        ).create_view()
        self.integrate_bind = self.integrate_shader.create_bind_group(self.match1d_view, linear_sampler)

    def push_passes_to(self, encoder: wgpu.GPUCommandEncoder):
        wgpu_util.push_3drender_pass(
            encoder, self.match1d_view, self.match1d_pipeline, self.match1d_bind
        )
        wgpu_util.push_2drender_pass(
            encoder, self.integrate_view, self.integrate_pipeline, self.integrate_bind
        )

    def draw(self) -> None:
        self.device.queue.submit(self.create_command_buffers())
