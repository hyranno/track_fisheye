
import cv2

import wgpu
from wgpu.gui.auto import WgpuCanvas, run
import wgpu.backends.rs  # noqa: F401, Select Rust backend
import numpy as np

from typing import Dict


def cvtype(mat: np.ndarray) -> int:
    cvdepth: Dict[str, int] = {
        "uint8": 0,
        "int8": 1,
        "uint16": 2,
        "int16": 3,
        "int32": 4,
        "float32": 5,
        "float64": 6,
    }
    depth: int = cvdepth[str(mat.dtype)]
    channels: int = mat.shape[2] - 1
    return depth * channels * 8


# Prepare WebGPU
canvas = WgpuCanvas(title="wgpu imageio")
adapter = wgpu.request_adapter(canvas=canvas, power_preference="high-performance")
device = adapter.request_device()
present_context = canvas.get_context()
render_texture_format = present_context.get_preferred_format(device.adapter)
present_context.configure(device=device, format=render_texture_format)

# read image file
img: np.ndarray = cv2.imread("resources/image.png", -1)

# write to GPUTexture
texture_data = img.data
texture_size: tuple[int, int, int] = img.shape[1], img.shape[0], 1
texture = device.create_texture(
    size=texture_size,
    usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.COPY_SRC | wgpu.TextureUsage.TEXTURE_BINDING,
    dimension=wgpu.TextureDimension.d2,
    format=wgpu.TextureFormat.rgba8unorm,
    mip_level_count=1,
    sample_count=1,
)
texture_data_layout = {
    "offset": 0,
    "bytes_per_row": img.shape[1] * img.shape[2],  # texture_data.strides[0],
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

# read from GPUTexture
dest_image_data: memoryview = device.queue.read_texture(
    texture_target,
    texture_data_layout,
    texture_size,
)
dest_image = np.asarray(dest_image_data).reshape(img.shape)

# write to file
# cv2.imwrite("imageio_out.png", dest_image)

# show on window
# see tests/test_rs_render_tex.py
shader_source = """
struct VertexInput {
    @builtin(vertex_index) vertex_index : u32,
};
struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@stage(vertex)
fn vs_main(in: VertexInput) -> VertexOutput {
    var positions = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0),
    );
    let index = i32(in.vertex_index);
    let p: vec2<f32> = positions[index];

    var out: VertexOutput;
    out.pos = vec4<f32>(p, 0.0, 1.0);
    out.uv = (-p + 1.0) / 2.0;
    return out;
}

@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var tex: texture_2d<f32>;
@stage(fragment)
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(tex, samp, in.uv);
}
"""
shader = device.create_shader_module(code=shader_source)

bind_group_layout = device.create_bind_group_layout(
    entries=[
        {
            "binding": 0,
            "visibility": wgpu.ShaderStage.FRAGMENT,
            "sampler": {
                "type": wgpu.SamplerBindingType.filtering,
            }
        },
        {
            "binding": 1,
            "visibility": wgpu.ShaderStage.FRAGMENT,
            "texture": {
                "sample_type": wgpu.TextureSampleType.float,
                "view_dimension": wgpu.TextureViewDimension.d2,
            },
        },
    ],
)
bind_group = device.create_bind_group(
    layout=bind_group_layout,
    entries=[
        {"binding": 0, "resource": device.create_sampler(min_filter="linear", mag_filter="linear")},
        {"binding": 1, "resource": texture.create_view(format=wgpu.TextureFormat.rgba8unorm)},
    ],
)

pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[bind_group_layout])
render_pipeline = device.create_render_pipeline(
    layout=pipeline_layout,
    vertex={
        "module": shader,
        "entry_point": "vs_main",
        "buffers": [],
    },
    primitive={
        "topology": wgpu.PrimitiveTopology.triangle_strip,
        "front_face": wgpu.FrontFace.ccw,
        "cull_mode": wgpu.CullMode.none,
    },
    depth_stencil=None,
    multisample=None,
    fragment={
        "module": shader,
        "entry_point": "fs_main",
        "targets": [
            {
                "format": render_texture_format,
                "blend": {
                    "color": (
                        wgpu.BlendFactor.one,
                        wgpu.BlendFactor.zero,
                        wgpu.BlendOperation.add,
                    ),
                    "alpha": (
                        wgpu.BlendFactor.one,
                        wgpu.BlendFactor.zero,
                        wgpu.BlendOperation.add,
                    ),
                },
            },
        ],
    },
)


def draw_frame():
    current_texture_view = present_context.get_current_texture()
    command_encoder = device.create_command_encoder()
    render_pass = command_encoder.begin_render_pass(
        color_attachments=[
            {
                "view": current_texture_view,
                "resolve_target": None,
                "clear_value": (0, 0, 0, 1),
                "load_op": wgpu.LoadOp.clear,
                "store_op": wgpu.StoreOp.store,
            }
        ],
    )

    render_pass.set_pipeline(render_pipeline)
    render_pass.set_bind_group(0, bind_group, [], 0, 0)
    render_pass.draw(4, 1, 0, 0)
    render_pass.end()
    device.queue.submit([command_encoder.finish()])


canvas.request_draw(draw_frame)


if __name__ == "__main__":
    run()
