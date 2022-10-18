import os
import wgpu
import numpy
import ctypes

# import mymodule
import wgpu_util
import texture_util


class Match1dMultiscaleShader(wgpu_util.RenderShaderBinding):
    def __init__(
        self,
        device: wgpu.GPUDevice,
        src_format: wgpu.TextureFormat,
    ):
        source = open(os.path.join(os.path.dirname(__file__), 'match1d_multiscale.wgsl'), "r").read()

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
                    "type": wgpu.BufferBindingType.read_only_storage,
                }
            },
            {
                "binding": 3,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "buffer": {
                    "type": wgpu.BufferBindingType.uniform,
                }
            },
            {
                "binding": 4,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "buffer": {
                    "type": wgpu.BufferBindingType.read_only_storage,
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
        kernel: numpy.ndarray,
        direction: tuple[float, float],
        scales: numpy.ndarray,
    ):
        buffer_kernel = self.device.create_buffer_with_data(
            data=kernel.astype(dtype=numpy.dtype('<f4'), order='C'),
            usage=wgpu.BufferUsage.STORAGE
        )
        buffer_direction = self.device.create_buffer_with_data(
            data=(ctypes.c_float * 2)(direction[0], direction[1]),
            usage=wgpu.BufferUsage.UNIFORM
        )
        buffer_scales = self.device.create_buffer_with_data(
            data=scales.astype(dtype=numpy.dtype('<u4'), order='C'),
            usage=wgpu.BufferUsage.STORAGE
        )
        entries = [
            {"binding": 0, "resource": src_view},
            {"binding": 1, "resource": src_sampler},
            {"binding": 2, "resource": {"buffer": buffer_kernel, "offset": 0, "size": buffer_kernel.size}},
            {"binding": 3, "resource": {"buffer": buffer_direction, "offset": 0, "size": buffer_direction.size}},
            {"binding": 4, "resource": {"buffer": buffer_scales, "offset": 0, "size": buffer_scales.size}},
        ]
        return super().create_bind_group(entries)
