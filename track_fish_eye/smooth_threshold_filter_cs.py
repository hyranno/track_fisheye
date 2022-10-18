import wgpu_util
import texture_util
import cv_util

import compute_shader.grayscale
import compute_shader.filter1d
import compute_shader.smooth_threshold

import sys
import array
import math
import numpy
import wgpu
from wgpu.gui.auto import run


def gaussian_kernel_ndarray(size: int) -> numpy.ndarray:
    kernel_data: array.array[float] = array.array('f')
    sigma: float = size / (2 * 3)
    for index in range(size):
        x = index - (size - 1) / 2
        kernel_data.append(math.exp(-x*x / (2*sigma*sigma)))
    denominator = sum(kernel_data)  # calc sum, not analytic integration, for approx error
    for index in range(size):
        kernel_data[index] /= denominator
    kernel = numpy.ndarray(
        shape=(1, size),
        dtype=numpy.float32,
        buffer=numpy.array(kernel_data)
    )
    return kernel


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

        self.grayscale_view = texture_util.create_buffer_texture(
            device, self.texture_size, wgpu.TextureFormat.rgba8snorm,
        ).create_view()
        self.grayscale_shader = compute_shader.grayscale.GrayscaleShader(device, src_view.texture.format)
        self.grayscale_pipeline = self.grayscale_shader.create_compute_pipeline()
        self.grayscale_bind = self.grayscale_shader.create_bind_group(
            src_view, self.grayscale_view
        )

        kernel_array = gaussian_kernel_ndarray(gaussian_kernel_size)
        self.gaussian_view = texture_util.create_buffer_texture(
            device, self.texture_size, wgpu.TextureFormat.rgba8snorm,
        ).create_view()
        self.gaussian_tmp_view = texture_util.create_buffer_texture(
            device, self.texture_size, wgpu.TextureFormat.rgba8snorm,
        ).create_view()
        self.gaussian_shader = compute_shader.filter1d.Filter1dShader(
            device,
            self.grayscale_view.texture.format,
        )
        self.gaussian_pipeline = self.gaussian_shader.create_compute_pipeline()
        self.gaussian_bind_horizontal = self.gaussian_shader.create_bind_group(
            self.grayscale_view, kernel_array, (1, 0), self.gaussian_tmp_view,
        )
        self.gaussian_bind_vertical = self.gaussian_shader.create_bind_group(
            self.gaussian_tmp_view, kernel_array, (0, 1), self.gaussian_view,
        )

        self.smooth_threshold_view = texture_util.create_buffer_texture(
            device, self.texture_size, wgpu.TextureFormat.rgba8snorm,
        ).create_view()
        self.smooth_threshold_shader = compute_shader.smooth_threshold.SmoothThresholdShader(
            device,
            self.gaussian_view.texture.format
        )
        self.smooth_threshold_pipeline = self.smooth_threshold_shader.create_compute_pipeline()
        self.smooth_threshold_bind = self.smooth_threshold_shader.create_bind_group(
            self.grayscale_view,
            self.gaussian_view,
            (-edge, edge),
            self.smooth_threshold_view,
        )

        return

    def push_compute_passes_to(self, encoder: wgpu.GPUCommandEncoder):
        wgpu_util.push_compute_pass(
            encoder, filter.grayscale_pipeline, filter.grayscale_bind,
            (filter.texture_size[0], filter.texture_size[1], 1)
        )
        wgpu_util.push_compute_pass(
            encoder, filter.gaussian_pipeline, filter.gaussian_bind_horizontal,
            (filter.texture_size[0], filter.texture_size[1], 1)
        )
        wgpu_util.push_compute_pass(
            encoder, filter.gaussian_pipeline, filter.gaussian_bind_vertical,
            (filter.texture_size[0], filter.texture_size[1], 1)
        )
        wgpu_util.push_compute_pass(
            encoder, filter.smooth_threshold_pipeline, filter.smooth_threshold_bind,
            (filter.texture_size[0], filter.texture_size[1], 1)
        )

    def compute(self) -> None:
        command_encoder = self.device.create_command_encoder()
        self.push_compute_passes_to(command_encoder)
        self.device.queue.submit([command_encoder.finish()])


if __name__ == "__main__":
    args = sys.argv
    # init image path from args or default
    src_path = "resources/image.png"
    print("processing " + src_path)
    device, context = wgpu_util.create_sized_canvas((512, 512), "image io example")
    context_texture_view = context.get_current_texture()
    context_texture_format = context.get_preferred_format(device.adapter)
    texture_src = cv_util.imread_texture(src_path, device)
    dest_view = texture_util.create_buffer_texture(device, texture_src.size).create_view()
    filter = SmoothThresholdFilter(
        device, texture_src.create_view(), dest_view, dest_view.texture.format
    )

    filter.compute()
    texture_util.draw_texture_on_texture(
        filter.smooth_threshold_view.texture, context_texture_view, context_texture_format, device
    )

    run()
