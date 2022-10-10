import os
import wgpu
from enum import Enum
import ctypes

# import mymodule
import wgpu_util
import texture_util


class OperatorMode(Enum):
    OR = 0
    NOR = 1
    AND = 2
    NAND = 3
    XOR = 4
    NXOR = 5


class BooleanOpsShader(wgpu_util.RenderShaderBinding):
    def __init__(
        self,
        device: wgpu.GPUDevice,
        src_format: wgpu.TextureFormat,
    ):
        header = open(os.path.join(os.path.dirname(__file__), 'header.wgsl'), "r").read()
        vs_source = open(os.path.join(os.path.dirname(__file__), 'vert_full2d.wgsl'), "r").read()
        fs_source = open(os.path.join(os.path.dirname(__file__), 'boolean_ops.wgsl'), "r").read()
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
                "texture": {
                    "sample_type": texture_util.texture_type_to_sample_type(src_format),
                    "view_dimension": wgpu.TextureViewDimension.d2,
                },
            },
            {
                "binding": 4,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "sampler": {
                    "type": wgpu.SamplerBindingType.filtering,
                }
            },
            {
                "binding": 5,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "buffer": {
                    "type": wgpu.BufferBindingType.uniform,
                }
            },
            {
                "binding": 6,
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
        view0: wgpu.GPUTextureView,
        sampler0: wgpu.GPUSampler,
        edge0: float,
        view1: wgpu.GPUTextureView,
        sampler1: wgpu.GPUSampler,
        edge1: float,
        op: OperatorMode,
    ):
        buffer_edge0 = self.device.create_buffer_with_data(
            data=(ctypes.c_float)(edge0),
            usage=wgpu.BufferUsage.UNIFORM
        )
        buffer_edge1 = self.device.create_buffer_with_data(
            data=(ctypes.c_float)(edge1),
            usage=wgpu.BufferUsage.UNIFORM
        )
        buffer_op = self.device.create_buffer_with_data(
            data=(ctypes.c_uint)(op.value),
            usage=wgpu.BufferUsage.UNIFORM
        )
        entries = [
            {"binding": 0, "resource": view0},
            {"binding": 1, "resource": sampler0},
            {"binding": 2, "resource": {"buffer": buffer_edge0, "offset": 0, "size": buffer_edge0.size}},
            {"binding": 3, "resource": view1},
            {"binding": 4, "resource": sampler1},
            {"binding": 5, "resource": {"buffer": buffer_edge1, "offset": 0, "size": buffer_edge1.size}},
            {"binding": 6, "resource": {"buffer": buffer_op, "offset": 0, "size": buffer_op.size}},
        ]
        return super().create_bind_group(entries)
