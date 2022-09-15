import os
import wgpu
from enum import Enum
import ctypes

# import mymodule
import wgpu_util
import texture_util


class NormalizerMode(Enum):
    NONE = 0
    NUM_SAMPLE = 1
    ONE_SUM = 2


class Filter1dShader(wgpu_util.RenderShaderBinding):
    def __init__(
        self,
        device: wgpu.GPUDevice,
        src_format: wgpu.TextureFormat,
        kernel_format: wgpu.TextureFormat,
        kernel_sampler_type: wgpu.SamplerBindingType,
    ):
        header = open(os.path.join(os.path.dirname(__file__), 'header.wgsl'), "r").read()
        vs_source = open(os.path.join(os.path.dirname(__file__), 'vert_full2d.wgsl'), "r").read()
        fs_source = open(os.path.join(os.path.dirname(__file__), 'filter_1d.wgsl'), "r").read()
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
                    "sample_type": texture_util.texture_type_to_sample_type(kernel_format),
                    "view_dimension": wgpu.TextureViewDimension.d1,
                },
            },
            {
                "binding": 4,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "sampler": {
                    "type": kernel_sampler_type,  # wgpu.SamplerBindingType.non_filtering,
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
        src_view: wgpu.GPUTextureView,
        src_sampler: wgpu.GPUSampler,
        resolution: tuple[int, int],
        kernel_view: wgpu.GPUTextureView,
        kernel_sampler: wgpu.GPUSampler,
        range: tuple[float, float],
        normalize_mode: NormalizerMode,
    ):
        buffer_resolution = self.device.create_buffer_with_data(
            data=(ctypes.c_uint32 * 2)(resolution[0], resolution[1]),
            usage=wgpu.BufferUsage.UNIFORM
        )
        buffer_range = self.device.create_buffer_with_data(
            data=(ctypes.c_float * 2)(range[0], range[1]),
            usage=wgpu.BufferUsage.UNIFORM
        )
        buffer_normalize_mode = self.device.create_buffer_with_data(
            data=(ctypes.c_uint32)(normalize_mode.value),
            usage=wgpu.BufferUsage.UNIFORM
        )
        entries = [
            {"binding": 0, "resource": src_view},
            {"binding": 1, "resource": src_sampler},
            {"binding": 2, "resource": {"buffer": buffer_resolution, "offset": 0, "size": buffer_resolution.size}},
            {"binding": 3, "resource": kernel_view},
            {"binding": 4, "resource": kernel_sampler},
            {"binding": 5, "resource": {"buffer": buffer_range, "offset": 0, "size": buffer_range.size}},
            {"binding": 6, "resource": {
                "buffer": buffer_normalize_mode, "offset": 0, "size": buffer_normalize_mode.size
            }},
        ]
        return super().create_bind_group(entries)
