import os
import wgpu
from enum import Enum
import ctypes

# import mymodule
import wgpu_util
import texture_util


class NegativeMode(Enum):
    NONE = 0
    ENABLE = 1


class ThresholdShader(wgpu_util.RenderShaderBinding):
    def __init__(
        self,
        device: wgpu.GPUDevice,
        src_format: wgpu.TextureFormat,
    ):
        header = open(os.path.join(os.path.dirname(__file__), 'header.wgsl'), "r").read()
        vs_source = open(os.path.join(os.path.dirname(__file__), 'vert_full2d.wgsl'), "r").read()
        fs_source = open(os.path.join(os.path.dirname(__file__), 'threshold.wgsl'), "r").read()
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
                    "type": wgpu.SamplerBindingType.filtering,
                }
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "buffer": {
                    "type": wgpu.BufferBindingType.uniform,
                }
            },
            {
                "binding": 3,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "buffer": {
                    "type": wgpu.BufferBindingType.uniform,
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
        view: wgpu.GPUTextureView,
        sampler: wgpu.GPUSampler,
        edge: float,
        negative: NegativeMode,
    ):
        buffer_edge = self.device.create_buffer_with_data(
            data=(ctypes.c_float)(edge),
            usage=wgpu.BufferUsage.UNIFORM
        )
        buffer_nega = self.device.create_buffer_with_data(
            data=(ctypes.c_uint)(negative.value),
            usage=wgpu.BufferUsage.UNIFORM
        )
        entries = [
            {"binding": 0, "resource": view},
            {"binding": 1, "resource": sampler},
            {"binding": 2, "resource": {"buffer": buffer_edge, "offset": 0, "size": buffer_edge.size}},
            {"binding": 3, "resource": {"buffer": buffer_nega, "offset": 0, "size": buffer_nega.size}},
        ]
        return super().create_bind_group(entries)
