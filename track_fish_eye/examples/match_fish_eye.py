import mymodule
import wgpu_util
import texture_util
import cv_util

import shader.match_multi_scale
import shader.match_result_integrate

import array
import math
import numpy
import wgpu
from wgpu.gui.auto import run


def push_2drender_pass(
    encoder: wgpu.GPUCommandEncoder,
    target: wgpu.GPUTextureView,
    pipeline: wgpu.GPURenderPipeline,
    bind: wgpu.GPUBindGroup,
):
    render_pass = encoder.begin_render_pass(
        color_attachments=[
            {
                "view": target,
                "resolve_target": None,
                "clear_value": (0, 0, 0, 1),
                "load_op": wgpu.LoadOp.clear,
                "store_op": wgpu.StoreOp.store,
            }
        ],
    )
    render_pass.set_pipeline(pipeline)
    render_pass.set_bind_group(0, bind, [], 0, 0)
    render_pass.draw(4, 1, 0, 0)
    render_pass.end()


def fisheye_kernel_ndarray() -> numpy.ndarray:
    kernel = numpy.array([
        1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0
    ], dtype=numpy.dtype('<f'), order='C')
    return kernel


def fisheye_PN_kernel_ndarray(kernel_half: numpy.ndarray) -> numpy.ndarray:
    kernel = numpy.concatenate((kernel_half, -kernel_half), axis=None)
    return kernel


def kernel1d_texture(device: wgpu.GPUDevice, kernel_ndarray: numpy.ndarray) -> wgpu.GPUTexture:
    texture_data = kernel_ndarray.data
    size = kernel_ndarray.size
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


resolution = 512, 512
device, context = wgpu_util.create_sized_canvas(resolution, "image io example")
context_texture_view = context.get_current_texture()
context_texture_format = context.get_preferred_format(device.adapter)

texture_src = cv_util.imread_texture("resources/image_preproced.png", device)
view_src = texture_src.create_view()

linear_sampler = device.create_sampler(min_filter="linear", mag_filter="linear")
nearest_sampler = device.create_sampler(min_filter="nearest", mag_filter="nearest")
tmp_texture = texture_util.create_buffer_texture(device, texture_src.size)
tmp_view = tmp_texture.create_view()

kernel = fisheye_kernel_ndarray()
match_view_vertical = texture_util.create_buffer_texture(device, texture_src.size).create_view()
match_view_horizontal = texture_util.create_buffer_texture(device, texture_src.size).create_view()
match_shader = shader.match_multi_scale.MatchMultiScale1dShader(device, texture_src.format)
match_pipeline = match_shader.create_render_pipeline([
    {
        "format": match_view_vertical.texture.format,
        "blend": texture_util.BLEND_STATE_REPLACE,
    },
])
match_threshold = 0.6
match_bind_vertical = match_shader.create_bind_group(
    view_src,
    linear_sampler,
    kernel,
    match_threshold,
    (0.0, 0.05),
    4.0,
    1.1,
)
match_bind_horizontal = match_shader.create_bind_group(
    view_src,
    linear_sampler,
    kernel,
    match_threshold,
    (0.05, 0.0),
    4.0,
    1.1,
)

match_result_view = texture_util.create_buffer_texture(device, match_view_vertical.texture.size).create_view()
match_result_shader = shader.match_result_integrate.MatchResultIntegrateShader(
    device, match_view_vertical.texture.format
)
match_result_pipeline = match_result_shader.create_render_pipeline([
    {
        "format": match_result_view.texture.format,
        "blend": texture_util.BLEND_STATE_REPLACE,
    },
])
match_result_bind = match_result_shader.create_bind_group(
    match_view_vertical,
    nearest_sampler,
    match_view_horizontal,
    nearest_sampler,
)


def draw(device: wgpu.GPUDevice):
    command_encoder = device.create_command_encoder()

    push_2drender_pass(command_encoder, match_view_horizontal, match_pipeline, match_bind_horizontal)
    push_2drender_pass(command_encoder, match_view_vertical, match_pipeline, match_bind_vertical)
    push_2drender_pass(command_encoder, match_result_view, match_result_pipeline, match_result_bind)

    device.queue.submit([command_encoder.finish()])


draw(device)
texture_util.draw_texture_on_texture(
    match_result_view.texture, context_texture_view, context_texture_format, device
)
shape = texture_src.size[1], texture_src.size[0], 4  # for numpy.ndarray
cv_util.imwrite_texture(match_result_view.texture, shape, "resources/image_matched.png", device)
run()
