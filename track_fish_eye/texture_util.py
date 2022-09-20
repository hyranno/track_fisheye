import wgpu
import wgpu.backends.rs  # noqa: F401, Select Rust backend

import shader.draw_texture

BLEND_STATE_REPLACE_SUB = (wgpu.BlendFactor.one, wgpu.BlendFactor.zero, wgpu.BlendOperation.add,)
BLEND_STATE_REPLACE = {
    "color": BLEND_STATE_REPLACE_SUB,
    "alpha": BLEND_STATE_REPLACE_SUB,
}


def texture_type_to_sample_type(texture_format: str) -> wgpu.TextureSampleType:
    if texture_format.endswith(("norm", "float")):
        texture_sample_type = wgpu.TextureSampleType.float
    elif "uint" in texture_format:
        texture_sample_type = wgpu.TextureSampleType.uint
    else:
        texture_sample_type = wgpu.TextureSampleType.sint
    return texture_sample_type


def create_buffer_texture(device: wgpu.GPUDevice, size: tuple[int, int, int]):
    return device.create_texture(
        size=size,
        usage=wgpu.TextureUsage.COPY_DST
        | wgpu.TextureUsage.COPY_SRC
        | wgpu.TextureUsage.TEXTURE_BINDING
        | wgpu.TextureUsage.RENDER_ATTACHMENT,
        dimension=wgpu.TextureDimension.d2,
        format=wgpu.TextureFormat.rgba8unorm,
        mip_level_count=1,
        sample_count=1,
    )


def create_buffer_texture_float(device: wgpu.GPUDevice, size: tuple[int, int, int]):
    return device.create_texture(
        size=size,
        usage=wgpu.TextureUsage.COPY_DST
        | wgpu.TextureUsage.COPY_SRC
        | wgpu.TextureUsage.TEXTURE_BINDING
        | wgpu.TextureUsage.RENDER_ATTACHMENT,
        dimension=wgpu.TextureDimension.d2,
        format=wgpu.TextureFormat.rgba32float,
        mip_level_count=1,
        sample_count=1,
    )


def draw_texture_on_texture(
    src: wgpu.GPUTexture, dest_view: wgpu.GPUTextureView, dest_format: wgpu.TextureFormat, device: wgpu.GPUDevice
):
    tex_shader = shader.draw_texture.TextureShader(device, src.format)
    pipeline = tex_shader.create_render_pipeline([
        {
            "format": dest_format,
            "blend": BLEND_STATE_REPLACE,
        },
    ])
    shader_bind = tex_shader.create_bind_group([
        {"binding": 0, "resource": device.create_sampler(min_filter="linear", mag_filter="linear")},
        {"binding": 1, "resource": src.create_view(format=src.format)},
    ])
    command_encoder = device.create_command_encoder()
    render_pass = command_encoder.begin_render_pass(
        color_attachments=[
            {
                "view": dest_view,  # dest.create_view(),
                "resolve_target": None,
                "clear_value": (0, 0, 0, 1),
                "load_op": wgpu.LoadOp.clear,
                "store_op": wgpu.StoreOp.store,
            }
        ],
    )
    render_pass.set_pipeline(pipeline)
    render_pass.set_bind_group(0, shader_bind, [], 0, 0)
    render_pass.draw(4, 1, 0, 0)
    render_pass.end()
    device.queue.submit([command_encoder.finish()])
