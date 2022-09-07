import wgpu
import wgpu.backends.rs  # noqa: F401, Select Rust backend


BLEND_STATE_REPLACE = (wgpu.BlendFactor.one, wgpu.BlendFactor.zero, wgpu.BlendOperation.add,)


def texture_type_to_sample_type(texture_format: str) -> wgpu.TextureSampleType:
    if texture_format.endswith(("norm", "float")):
        texture_sample_type = wgpu.TextureSampleType.float
    elif "uint" in texture_format:
        texture_sample_type = wgpu.TextureSampleType.uint
    else:
        texture_sample_type = wgpu.TextureSampleType.sint
    return texture_sample_type


# def draw_texture_on_texture(src: wgpu.GPUTexture, dest: wgpu.GPUTexture, device: wgpu.GPUDevice):
def draw_texture_on_texture(
    src: wgpu.GPUTexture, dest_view: wgpu.GPUTextureView, dest_format: wgpu.TextureFormat, device: wgpu.GPUDevice
):
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
                    "sample_type": texture_type_to_sample_type(src.format),
                    "view_dimension": wgpu.TextureViewDimension.d2,
                },
            },
        ],
    )
    bind_group = device.create_bind_group(
        layout=bind_group_layout,
        entries=[
            {"binding": 0, "resource": device.create_sampler(min_filter="linear", mag_filter="linear")},
            {"binding": 1, "resource": src.create_view(format=src.format)},
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
                    "format": dest_format,
                    "blend": {
                        "color": BLEND_STATE_REPLACE,
                        "alpha": BLEND_STATE_REPLACE,
                    },
                },
            ],
        },
    )

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

    render_pass.set_pipeline(render_pipeline)
    render_pass.set_bind_group(0, bind_group, [], 0, 0)
    render_pass.draw(4, 1, 0, 0)
    render_pass.end()
    device.queue.submit([command_encoder.finish()])
