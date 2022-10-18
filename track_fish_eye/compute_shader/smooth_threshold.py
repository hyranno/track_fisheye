import os
import wgpu
import ctypes

# import mymodule
import wgpu_util
import texture_util


class SmoothThresholdShader(wgpu_util.ComputeShaderBinding):
    def __init__(
        self,
        device: wgpu.GPUDevice,
        src_format: wgpu.TextureFormat,
    ):
        header = open(os.path.join(os.path.dirname(__file__), 'util.wgsl'), "r").read()
        cs_source = open(os.path.join(os.path.dirname(__file__), 'smooth_threshold.wgsl'), "r").read()
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
                "binding": 1,  # mean
                "visibility": wgpu.ShaderStage.COMPUTE,
                "texture": {
                    "sample_type": texture_util.texture_type_to_sample_type(src_format),
                    "view_dimension": wgpu.TextureViewDimension.d2,
                },
            },
            {
                "binding": 2,  # edges
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
        src: wgpu.GPUTextureView,
        mean: wgpu.GPUTextureView,
        edges: tuple[float, float],
        dest: wgpu.GPUTextureView,
    ):
        buffer_edges = self.device.create_buffer_with_data(
            data=(ctypes.c_float * 2)(edges[0], edges[1]),
            usage=wgpu.BufferUsage.UNIFORM
        )
        entries = [
            {"binding": 0, "resource": src},
            {"binding": 1, "resource": mean},
            {"binding": 2, "resource": {"buffer": buffer_edges, "offset": 0, "size": buffer_edges.size}},
            {"binding": 3, "resource": dest},
        ]
        return super().create_bind_group(entries)
