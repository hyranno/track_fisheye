import wgpu_util
import texture_util
import cv_util

import shader.grayscale
import shader.filter_1d
import shader.smooth_threshold

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


def kernel1d_texture(device: wgpu.GPUDevice, kernel: numpy.ndarray) -> wgpu.GPUTexture:
    size = kernel.shape[1]
    texture_data = kernel.data
    texture_size: tuple[int, int, int] = size, 1, 1
    texture = device.create_texture(
        size=texture_size,
        usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.COPY_SRC | wgpu.TextureUsage.TEXTURE_BINDING,
        dimension=wgpu.TextureDimension.d1,
        format=wgpu.TextureFormat.r32float,
        mip_level_count=1,
        sample_count=1,
    )
    texture_data_layout = {
        "offset": 0,
        "bytes_per_row": 4 * size,  # texture_data.strides[0],
        "rows_per_image": 0,
    }
    texture_target = {
        "texture": texture,
        "mip_level": 0,
        "origin": (0, 0, 0),
    }
    device.queue.write_texture(
        texture_target,
        texture_data,
        texture_data_layout,
        texture_size,
    )
    return texture


class SmoothThresholdFilter:
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
        nearest_sampler = device.create_sampler(min_filter="nearest", mag_filter="nearest")

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
        self.gaussian_kernel_size = 79
        # one big kernel will be efficient rather than repeated blur
        # total_kernel_size = sqrt(num_loop) * kernel_size
        kernel_array = gaussian_kernel_ndarray(self.gaussian_kernel_size)
        self.gaussian_kernel_view = kernel1d_texture(device, kernel_array).create_view()
        self.gaussian_shader = shader.filter_1d.Filter1dShader(
            device,
            self.grayscale_view.texture.format,
            self.gaussian_kernel_view.texture.format,
            wgpu.SamplerBindingType.filtering
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
            self.gaussian_kernel_view,
            linear_sampler,
            (self.gaussian_kernel_size / self.texture_size[0], 0.0),
            shader.filter_1d.NormalizerMode.ONE_SUM,
        )
        self.gaussian_bind_vertical = self.gaussian_shader.create_bind_group(
            self.gaussian_tmp_view,
            linear_sampler,
            self.gaussian_kernel_view,
            linear_sampler,
            (0.0, self.gaussian_kernel_size / self.texture_size[1]),
            shader.filter_1d.NormalizerMode.ONE_SUM,
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
            (-0.1, 0.1)
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
    filter = SmoothThresholdFilter(device, texture_src.create_view(), dest_view.texture.format)

    filter.draw(dest_view)
    texture_util.draw_texture_on_texture(dest_view.texture, context_texture_view, context_texture_format, device)
    run()
