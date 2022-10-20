import os
import wgpu
import numpy

# import mymodule
import wgpu_util
import texture_util


class Match1d2dIntegrateShader(wgpu_util.ComputeShaderBinding):
    def __init__(
        self,
        device: wgpu.GPUDevice,
        src_format: wgpu.TextureFormat,
    ):
        header = open(os.path.join(os.path.dirname(__file__), 'util.wgsl'), "r").read()
        cs_source = open(os.path.join(os.path.dirname(__file__), 'match1d2d_integrate.wgsl'), "r").read()
        source = header + cs_source
        bind_entries = [
            {
                "binding": 0,  # src0
                "visibility": wgpu.ShaderStage.COMPUTE,
                "texture": {
                    "sample_type": texture_util.texture_type_to_sample_type(src_format),
                    "view_dimension": wgpu.TextureViewDimension.d2,
                },
            },
            {
                "binding": 1,  # src1
                "visibility": wgpu.ShaderStage.COMPUTE,
                "texture": {
                    "sample_type": texture_util.texture_type_to_sample_type(src_format),
                    "view_dimension": wgpu.TextureViewDimension.d2,
                },
            },
            {
                "binding": 2,  # threshold
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
                    "format": wgpu.TextureFormat.rgba8unorm,
                    "view_dimension": wgpu.TextureViewDimension.d2,
                },
            },
        ]
        super().__init__(device, source, "main", bind_entries)

    def create_bind_group(
        self,
        src0: wgpu.GPUTextureView,
        src1: wgpu.GPUTextureView,
        threshold: float,
        dest: wgpu.GPUTextureView,
    ):
        buffer_threshold = self.device.create_buffer_with_data(
            data=numpy.array([threshold], dtype=numpy.dtype('<f'), order='C'),
            usage=wgpu.BufferUsage.UNIFORM
        )
        entries = [
            {"binding": 0, "resource": src0},
            {"binding": 1, "resource": src1},
            {"binding": 2, "resource": {"buffer": buffer_threshold, "offset": 0, "size": buffer_threshold.size}},
            {"binding": 3, "resource": dest},
        ]
        return super().create_bind_group(entries)
