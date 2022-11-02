import os
import wgpu

# import mymodule
import wgpu_util
import texture_util


class TextureShader(wgpu_util.RenderShaderBinding):
    def __init__(
        self,
        device: wgpu.GPUDevice,
        src_format: wgpu.TextureFormat,
        sampler_type: wgpu.SamplerBindingType = wgpu.SamplerBindingType.filtering,
    ):
        header = open(os.path.join(os.path.dirname(__file__), 'header.wgsl'), "r").read()
        vs_source = open(os.path.join(os.path.dirname(__file__), 'vert_full2d.wgsl'), "r").read()
        fs_source = open(os.path.join(os.path.dirname(__file__), 'draw_texture.wgsl'), "r").read()
        source = header + vs_source + fs_source
        bind_entries = [
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "texture": {
                    "sample_type": texture_util.texture_type_to_sample_type(src_format),
                    "view_dimension": wgpu.TextureViewDimension.d2,
                },
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "sampler": {
                    "type": sampler_type,
                }
            },
        ]
        vertex = {
            "entry_point": "vs_main",
            "buffers": [],
        }
        primitive = {
            "topology": wgpu.PrimitiveTopology.triangle_strip,
            "front_face": wgpu.FrontFace.ccw,
            "cull_mode": wgpu.CullMode.none,
        }
        fragment = {
            "entry_point": "fs_main",
        }
        super().__init__(device, source, bind_entries, vertex, primitive, fragment)

    def create_bind_group(
        self,
        src_view: wgpu.GPUTextureView,
        src_sampler: wgpu.GPUSampler,
    ):
        entries = [
            {"binding": 0, "resource": src_view},
            {"binding": 1, "resource": src_sampler},
        ]
        return super().create_bind_group(entries)
