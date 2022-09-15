import mymodule
import wgpu_util
import texture_util
import cv_util

import shader.grayscale
import shader.filter_1d
import shader.smooth_threshold

import array
import math
import numpy
import wgpu
from wgpu.gui.auto import run
import ctypes


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


def gaussian_kernel_texture(size: int, device: wgpu.GPUDevice) -> wgpu.GPUTexture:
    texture_data = gaussian_kernel_ndarray(size).data
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


device, context = wgpu_util.create_sized_canvas((512, 512), "image io example")
context_texture_view = context.get_current_texture()
context_texture_format = context.get_preferred_format(device.adapter)

texture_src = cv_util.imread_texture("resources/image.png", device)
shape = texture_src.size[1], texture_src.size[0], 4  # for numpy.ndarray
linear_sampler = device.create_sampler(min_filter="linear", mag_filter="linear")
nearest_sampler = device.create_sampler(min_filter="nearest", mag_filter="nearest")
horizontal_tmp_texture = texture_util.create_buffer_texture(device, texture_src.size)
horizontal_tmp_view = horizontal_tmp_texture.create_view()

grayscale_texture = texture_util.create_buffer_texture(device, texture_src.size)
grayscale_view = grayscale_texture.create_view()
grayscale_shader = shader.grayscale.GrayscaleShader(device, texture_src.format)
grayscale_pipeline = grayscale_shader.create_render_pipeline([
    {
        "format": grayscale_texture.format,  # context_texture_format,
        "blend": texture_util.BLEND_STATE_REPLACE,
    },
])
grayscale_bind = grayscale_shader.create_bind_group([
    {"binding": 0, "resource": linear_sampler},
    {"binding": 1, "resource": texture_src.create_view(format=texture_src.format)},
])

uniform_resolution = device.create_buffer_with_data(
    data=(ctypes.c_uint32 * 2)(512, 512),
    usage=wgpu.BufferUsage.UNIFORM
)

gaussian_texture = texture_util.create_buffer_texture(device, texture_src.size)
gaussian_view = gaussian_texture.create_view()
gaussian_kernel_size = 11
gaussian_kernel = gaussian_kernel_texture(gaussian_kernel_size, device)
gaussian_shader = shader.filter_1d.Filter1dShader(
    device,
    grayscale_texture.format,
    gaussian_kernel.format,
    wgpu.SamplerBindingType.filtering
)
gaussian_pipeline = gaussian_shader.create_render_pipeline([
    {
        "format": grayscale_texture.format,
        "blend": texture_util.BLEND_STATE_REPLACE,
    },
])
gaussian_normalize = shader.filter_1d.NormalizerMode.create_uniform_buffer(
    device,
    shader.filter_1d.NormalizerMode.ONE_SUM
)
gaussian_uniform_horizontal = device.create_buffer_with_data(
    data=(ctypes.c_float * 2)(gaussian_kernel_size / 512.0, 0.0),
    usage=wgpu.BufferUsage.UNIFORM
)
gaussian_bind_horizontal = gaussian_shader.create_bind_group([
    {"binding": 0, "resource": grayscale_view},
    {"binding": 1, "resource": linear_sampler},
    {"binding": 2, "resource": {"buffer": uniform_resolution, "offset": 0, "size": uniform_resolution.size}},
    {"binding": 3, "resource": gaussian_kernel.create_view(format=gaussian_kernel.format)},
    {"binding": 4, "resource": linear_sampler},
    {"binding": 5, "resource": {
        "buffer": gaussian_uniform_horizontal,
        "offset": 0,
        "size": gaussian_uniform_horizontal.size
    }},
    {"binding": 6, "resource": {"buffer": gaussian_normalize, "offset": 0, "size": gaussian_normalize.size}},
])
gaussian_uniform_vertical = device.create_buffer_with_data(
    data=(ctypes.c_float * 2)(0.0, gaussian_kernel_size / 512.0),
    usage=wgpu.BufferUsage.UNIFORM
)
gaussian_bind_vertical = gaussian_shader.create_bind_group([
    {"binding": 0, "resource": horizontal_tmp_view},
    {"binding": 1, "resource": linear_sampler},
    {"binding": 2, "resource": {"buffer": uniform_resolution, "offset": 0, "size": uniform_resolution.size}},
    {"binding": 3, "resource": gaussian_kernel.create_view(format=gaussian_kernel.format)},
    {"binding": 4, "resource": linear_sampler},
    {"binding": 5, "resource": {
        "buffer": gaussian_uniform_vertical,
        "offset": 0,
        "size": gaussian_uniform_vertical.size
    }},
    {"binding": 6, "resource": {"buffer": gaussian_normalize, "offset": 0, "size": gaussian_normalize.size}},
])

smooth_threshold_shader = shader.smooth_threshold.SmoothThresholdShader(device, texture_src.format)
smooth_edges = device.create_buffer_with_data(
    data=(ctypes.c_float * 2)(-0.2, 0.2),
    usage=wgpu.BufferUsage.UNIFORM
)
smooth_threshold_pipeline = smooth_threshold_shader.create_render_pipeline([
    {
        "format": context_texture_format,  # context_texture_format,
        "blend": texture_util.BLEND_STATE_REPLACE,
    },
])
smooth_threshold_bind = smooth_threshold_shader.create_bind_group([
    {"binding": 0, "resource": linear_sampler},
    {"binding": 1, "resource": grayscale_view},
    {"binding": 2, "resource": gaussian_view},
    {"binding": 3, "resource": {"buffer": smooth_edges, "offset": 0, "size": smooth_edges.size}},
])


def draw(
    src: wgpu.GPUTexture, dest_view: wgpu.GPUTextureView, dest_format: wgpu.TextureFormat, device: wgpu.GPUDevice
):
    command_encoder = device.create_command_encoder()

    render_pass = command_encoder.begin_render_pass(
        color_attachments=[
            {
                "view": grayscale_view,  # TODO causing bug
                "resolve_target": None,
                "clear_value": (0, 0, 0, 1),
                "load_op": wgpu.LoadOp.clear,
                "store_op": wgpu.StoreOp.store,
            }
        ],
    )
    render_pass.set_pipeline(grayscale_pipeline)
    render_pass.set_bind_group(0, grayscale_bind, [], 0, 0)
    render_pass.draw(4, 1, 0, 0)
    render_pass.end()

    render_pass = command_encoder.begin_render_pass(
        color_attachments=[
            {
                "view": horizontal_tmp_view,
                "resolve_target": None,
                "clear_value": (0, 0, 0, 1),
                "load_op": wgpu.LoadOp.clear,
                "store_op": wgpu.StoreOp.store,
            }
        ],
    )
    render_pass.set_pipeline(gaussian_pipeline)
    render_pass.set_bind_group(0, gaussian_bind_horizontal, [], 0, 0)
    render_pass.draw(4, 1, 0, 0)
    render_pass.end()

    render_pass = command_encoder.begin_render_pass(
        color_attachments=[
            {
                "view": gaussian_view,  # dest.create_view(),
                "resolve_target": None,
                "clear_value": (0, 0, 0, 1),
                "load_op": wgpu.LoadOp.clear,
                "store_op": wgpu.StoreOp.store,
            }
        ],
    )
    render_pass.set_pipeline(gaussian_pipeline)
    render_pass.set_bind_group(0, gaussian_bind_vertical, [], 0, 0)
    render_pass.draw(4, 1, 0, 0)
    render_pass.end()

    render_pass = command_encoder.begin_render_pass(
        color_attachments=[
            {
                "view": context_texture_view,  # dest.create_view(),
                "resolve_target": None,
                "clear_value": (0, 0, 0, 1),
                "load_op": wgpu.LoadOp.clear,
                "store_op": wgpu.StoreOp.store,
            }
        ],
    )
    render_pass.set_pipeline(smooth_threshold_pipeline)
    render_pass.set_bind_group(0, smooth_threshold_bind, [], 0, 0)
    render_pass.draw(4, 1, 0, 0)
    render_pass.end()

    device.queue.submit([command_encoder.finish()])


draw(texture_src, context_texture_view, context_texture_format, device)
run()
