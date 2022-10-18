import os
import wgpu
import ctypes
import numpy

# import mymodule
import wgpu_util
import texture_util


class Filter1dShader(wgpu_util.ComputeShaderBinding):
    def __init__(
        self,
        device: wgpu.GPUDevice,
        src_format: wgpu.TextureFormat,
    ):
        header = open(os.path.join(os.path.dirname(__file__), 'util.wgsl'), "r").read()
        cs_source = open(os.path.join(os.path.dirname(__file__), 'filter1d.wgsl'), "r").read()
        source = header + cs_source

        bind_entries = [
            {
                "binding": 0,  # src
                "visibility": wgpu.ShaderStage.COMPUTE,
                "texture": {
                    "sample_type": texture_util.texture_type_to_sample_type(src_format),
                    "view_dimension": wgpu.TextureViewDimension.d2,
                },
            },
            {
                "binding": 1,  # kernel
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {
                    "type": wgpu.BufferBindingType.read_only_storage,
                }
            },
            {
                "binding": 2,  # direction
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {
                    "type": wgpu.BufferBindingType.uniform,
                }
            },
            {
                "binding": 3,  # dest
                "visibility": wgpu.ShaderStage.COMPUTE,
                "storage_texture": {
                    "access": wgpu.StorageTextureAccess.write_only,
                    "format": wgpu.TextureFormat.rgba8snorm,
                    "view_dimension": wgpu.TextureViewDimension.d2,
                },
            },
        ]
        super().__init__(device, source, "main", bind_entries)

    def create_bind_group(
        self,
        src_view: wgpu.GPUTextureView,
        kernel: numpy.ndarray,
        direction: tuple[int, int],
        dest_view: wgpu.GPUTextureView,
    ):
        buffer_kernel = self.device.create_buffer_with_data(
            data=kernel.astype(dtype=numpy.dtype('<f'), order='C'),
            usage=wgpu.BufferUsage.STORAGE
        )
        buffer_direction = self.device.create_buffer_with_data(
            data=(ctypes.c_int32 * 2)(direction[0], direction[1]),
            usage=wgpu.BufferUsage.UNIFORM
        )
        entries = [
            {"binding": 0, "resource": src_view},
            {"binding": 1, "resource": {"buffer": buffer_kernel, "offset": 0, "size": buffer_kernel.size}},
            {"binding": 2, "resource": {"buffer": buffer_direction, "offset": 0, "size": buffer_direction.size}},
            {"binding": 3, "resource": dest_view},
        ]
        return super().create_bind_group(entries)
