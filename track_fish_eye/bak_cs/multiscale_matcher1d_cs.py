import wgpu_util
import texture_util
import cv_util

from compute_shader.match1d_multiscale import Match1dMultiscaleShader
from compute_shader.minmaxmean import MinMaxMeanShader

import sys
import numpy
import wgpu
from wgpu.gui.auto import run


class MultiScaleMatcher1d:
    def __init__(
        self,
        device: wgpu.GPUDevice,
        src_view: wgpu.GPUTextureView,
        dest_view: wgpu.GPUTextureView,  # rgba8snorm
        kernel: numpy.ndarray,
        direction: tuple[int, int],
        scale_min: int,
        scale_max: int,
        scale_step: float,
    ):
        self.device = device
        self.texture_size = src_view.texture.size

        self.scales = []
        s = float(scale_min)
        while (s <= scale_max):
            self.scales.append(int(s))
            s *= scale_step

        self.match1d_shader = Match1dMultiscaleShader(device, src_view.texture.format)
        self.match1d_pipeline = self.match1d_shader.create_compute_pipeline()
        self.match1d_view = device.create_texture(
            size=(self.texture_size[0], self.texture_size[1], len(self.scales)),
            usage=wgpu.TextureUsage.COPY_DST
            | wgpu.TextureUsage.COPY_SRC
            | wgpu.TextureUsage.TEXTURE_BINDING
            | wgpu.TextureUsage.STORAGE_BINDING,
            dimension=wgpu.TextureDimension.d3,
            format=wgpu.TextureFormat.rgba8snorm,
            mip_level_count=1,
            sample_count=1,
        ).create_view()
        self.match1d_bind = self.match1d_shader.create_bind_group(
            src_view, kernel, direction, numpy.array(self.scales), self.match1d_view
        )

        self.integrate_shader = MinMaxMeanShader(device, self.match1d_view.texture.format)
        self.integrate_pipeline = self.integrate_shader.create_compute_pipeline()
        self.integrate_bind = self.integrate_shader.create_bind_group(self.match1d_view, dest_view)

    def push_compute_passes_to(self, encoder: wgpu.GPUCommandEncoder):
        wgpu_util.push_compute_pass(
            encoder, self.match1d_pipeline, self.match1d_bind,
            (self.texture_size[0], self.texture_size[1], len(self.scales))
        )
        wgpu_util.push_compute_pass(
            encoder, self.integrate_pipeline, self.integrate_bind,
            (self.texture_size[0], self.texture_size[1], 1)
        )

    def compute(self) -> None:
        command_encoder = self.device.create_command_encoder()
        self.push_compute_passes_to(command_encoder)
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
    dest_view = texture_util.create_buffer_texture(
        device, texture_src.size, wgpu.TextureFormat.rgba8snorm
    ).create_view()
    kernel = numpy.array([
        1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0
    ], dtype=numpy.dtype('<f'), order='C')
    filter = MultiScaleMatcher1d(
        device, texture_src.create_view(), dest_view,
        kernel, (1, 0), 20, 200, 1.1
    )

    filter.compute()
    texture_util.draw_texture_on_texture(
        dest_view.texture, context_texture_view, context_texture_format, device
    )

    run()
