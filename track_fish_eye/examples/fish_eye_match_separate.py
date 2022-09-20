import mymodule
import wgpu_util
import texture_util
import cv_util

import shader.remap
import shader.filter_1d
import shader.threshold
import shader.boolean_ops

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


resolution = 512, 512
device, context = wgpu_util.create_sized_canvas(resolution, "image io example")
context_texture_view = context.get_current_texture()
context_texture_format = context.get_preferred_format(device.adapter)

texture_src = cv_util.imread_texture("resources/image_preproced.png", device)

linear_sampler = device.create_sampler(min_filter="linear", mag_filter="linear")
nearest_sampler = device.create_sampler(min_filter="nearest", mag_filter="nearest")
tmp_texture = texture_util.create_buffer_texture(device, texture_src.size)
tmp_view = tmp_texture.create_view()

remap_view = texture_util.create_buffer_texture_float(device, texture_src.size).create_view()
remap_shader = shader.remap.RemapShader(device, texture_src.format)
remap_pipeline = remap_shader.create_render_pipeline([
    {
        "format": remap_view.texture.format,  # context_texture_format,
        "blend": texture_util.BLEND_STATE_REPLACE,
    },
])
remap_bind = remap_shader.create_bind_group(
    texture_src.create_view(),
    linear_sampler,
    (-0.5, -0.5, -0.5, 0.0),
    (2.0, 2.0, 2.0, 1.0),
)

threshold_view_positive = texture_util.create_buffer_texture(device, texture_src.size).create_view()
threshold_view_negative = texture_util.create_buffer_texture(device, texture_src.size).create_view()
threshold_shader = shader.threshold.ThresholdShader(device, remap_view.texture.format)
threshold_pipeline = threshold_shader.create_render_pipeline([
    {
        "format": threshold_view_positive.texture.format,
        "blend": texture_util.BLEND_STATE_REPLACE,
    },
])
threshold_bind_positive = threshold_shader.create_bind_group(
    remap_view,
    linear_sampler,
    0.2,
    shader.threshold.NegativeMode.NONE,
)
threshold_bind_negative = threshold_shader.create_bind_group(
    remap_view,
    linear_sampler,
    -0.2,
    shader.threshold.NegativeMode.ENABLE,
)

bool_view = texture_util.create_buffer_texture(device, texture_src.size).create_view()
bool_shader = shader.boolean_ops.BooleanOpsShader(device, threshold_view_positive.texture.format)
bool_pipeline = bool_shader.create_render_pipeline([
    {
        "format": bool_view.texture.format,
        "blend": texture_util.BLEND_STATE_REPLACE,
    },
])
bool_bind = bool_shader.create_bind_group(
    threshold_view_positive,
    linear_sampler,
    0.5,
    threshold_view_negative,
    linear_sampler,
    0.5,
    shader.boolean_ops.OperatorMode.OR,
)


def draw(
    src: wgpu.GPUTexture, dest_view: wgpu.GPUTextureView, dest_format: wgpu.TextureFormat, device: wgpu.GPUDevice
):
    command_encoder = device.create_command_encoder()

    push_2drender_pass(command_encoder, remap_view, remap_pipeline, remap_bind)
    push_2drender_pass(command_encoder, threshold_view_positive, threshold_pipeline, threshold_bind_positive)
    push_2drender_pass(command_encoder, threshold_view_negative, threshold_pipeline, threshold_bind_negative)
    push_2drender_pass(command_encoder, bool_view, bool_pipeline, bool_bind)

    device.queue.submit([command_encoder.finish()])


draw(texture_src, context_texture_view, context_texture_format, device)
texture_util.draw_texture_on_texture(
    bool_view.texture, context_texture_view, context_texture_format, device
)
run()
