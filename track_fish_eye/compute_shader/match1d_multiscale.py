import os
import wgpu
import numpy
import ctypes

# import mymodule
import wgpu_util
import texture_util


class Match1dMultiscaleShader(wgpu_util.ComputeShaderBinding):
    def __init__(
        self,
        device: wgpu.GPUDevice,
        src_format: wgpu.TextureFormat,
    ):
        header = open(os.path.join(os.path.dirname(__file__), 'util.wgsl'), "r").read()
        cs_source = open(os.path.join(os.path.dirname(__file__), 'match1d_multiscale.wgsl'), "r").read()
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
                "binding": 3,  # scales_px
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
                    "format": wgpu.TextureFormat.rgba8snorm,
                    "view_dimension": wgpu.TextureViewDimension.d3,
                },
            },
        ]
        super().__init__(device, source, "main", bind_entries)

    def create_bind_group(
        self,
        src: wgpu.GPUTextureView,
        kernel: numpy.ndarray,
        direction: tuple[int, int],
        scales: numpy.ndarray,
        dest: wgpu.GPUTextureView,
    ):
        buffer_kernel = self.device.create_buffer_with_data(
            data=kernel.astype(dtype=numpy.dtype('<f4'), order='C'),
            usage=wgpu.BufferUsage.STORAGE
        )
        buffer_direction = self.device.create_buffer_with_data(
            data=numpy.array(direction).astype(dtype=numpy.dtype('<i4'), order='C'),
            usage=wgpu.BufferUsage.UNIFORM
        )
        buffer_scales = self.device.create_buffer_with_data(
            data=scales.astype(dtype=numpy.dtype('<u4'), order='C'),
            usage=wgpu.BufferUsage.STORAGE
        )
        entries = [
            {"binding": 0, "resource": src},
            {"binding": 1, "resource": {"buffer": buffer_kernel, "offset": 0, "size": buffer_kernel.size}},
            {"binding": 2, "resource": {"buffer": buffer_direction, "offset": 0, "size": buffer_direction.size}},
            {"binding": 3, "resource": {"buffer": buffer_scales, "offset": 0, "size": buffer_scales.size}},
            {"binding": 4, "resource": dest},
        ]
        return super().create_bind_group(entries)
