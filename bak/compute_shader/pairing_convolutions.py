import os
import wgpu
import ctypes
import numpy

# import mymodule
import wgpu_util
import texture_util


class PairingConvolutionShader(wgpu_util.ComputeShaderBinding):
    def __init__(
        self,
        device: wgpu.GPUDevice,
        src_format: wgpu.TextureFormat,
    ):
        header = open(os.path.join(os.path.dirname(__file__), 'util.wgsl'), "r").read()
        cs_source = open(os.path.join(os.path.dirname(__file__), 'pairing_convolutions.wgsl'), "r").read()
        source = header + cs_source

        bind_layout_entries = [
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "texture": {
                    "sample_type": texture_util.texture_type_to_sample_type(src_format),
                    "view_dimension": wgpu.TextureViewDimension.d2,
                },
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {
                    "type": wgpu.BufferBindingType.read_only_storage,
                }
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {
                    "type": wgpu.BufferBindingType.read_only_storage,
                }
            },
            {
                "binding": 3,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {
                    "type": wgpu.BufferBindingType.read_only_storage,
                }
            },
            {
                "binding": 4,  # dest
                "visibility": wgpu.ShaderStage.COMPUTE,
                "storage_texture": {
                    "access": wgpu.StorageTextureAccess.write_only,
                    "format": wgpu.TextureFormat.rgba8unorm,
                    "view_dimension": wgpu.TextureViewDimension.d2,
                },
            },
        ]
        super().__init__(device, source, "main", bind_layout_entries)

    def create_bind_group(
        self,
        src: wgpu.GPUTextureView,
        kernel: numpy.ndarray,
        starts: numpy.ndarray,
        ends: numpy.ndarray,
        dest: wgpu.GPUTextureView,
    ) -> wgpu.GPUBindGroup:
        buffer_kernel = self.device.create_buffer_with_data(
            data=kernel.astype(dtype=numpy.dtype('<f'), order='C'),
            usage=wgpu.BufferUsage.STORAGE
        )
        buffer_starts = self.device.create_buffer_with_data(
            data=starts.astype(dtype=numpy.dtype('<f'), order='C'),
            usage=wgpu.BufferUsage.STORAGE
        )
        buffer_ends = self.device.create_buffer_with_data(
            data=ends.astype(dtype=numpy.dtype('<f'), order='C'),
            usage=wgpu.BufferUsage.STORAGE
        )
        entries = [
            {"binding": 0, "resource": src},
            {"binding": 1, "resource": {"buffer": buffer_kernel, "offset": 0, "size": buffer_kernel.size}},
            {"binding": 2, "resource": {"buffer": buffer_starts, "offset": 0, "size": buffer_starts.size}},
            {"binding": 3, "resource": {"buffer": buffer_ends, "offset": 0, "size": buffer_ends.size}},
            {"binding": 4, "resource": dest},
        ]

        return super().create_bind_group(entries)
