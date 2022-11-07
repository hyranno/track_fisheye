import os
import wgpu
import numpy
import ctypes

import wgpu_util
import texture_util


class Filter1dShader(wgpu_util.RenderShader2d):
    def __init__(
        self,
        device: wgpu.GPUDevice,
        src_format: wgpu.TextureFormat,
    ):
        fs_source = open(os.path.join(os.path.dirname(__file__), 'filter1d.wgsl'), "r").read()
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
        ]
        super().__init__(device, fs_source, bind_entries)

    def create_bind_group(
        self,
        src_view: wgpu.GPUTextureView,
        src_sampler: wgpu.GPUSampler,
        kernel: numpy.ndarray,
        direction: tuple[int, int],
    ):
        buffer_kernel = self.device.create_buffer_with_data(
            data=kernel.astype(dtype=numpy.dtype('<f'), order='C'),
            usage=wgpu.BufferUsage.STORAGE
        )
        buffer_direction = self.device.create_buffer_with_data(
            data=(ctypes.c_int * 2)(direction[0], direction[1]),
            usage=wgpu.BufferUsage.UNIFORM
        )
        entries = [
            {"binding": 0, "resource": src_view},
            {"binding": 1, "resource": src_sampler},
            {"binding": 2, "resource": {"buffer": buffer_kernel, "offset": 0, "size": buffer_kernel.size}},
            {"binding": 3, "resource": {"buffer": buffer_direction, "offset": 0, "size": buffer_direction.size}},
        ]
        return super().create_bind_group(entries)
