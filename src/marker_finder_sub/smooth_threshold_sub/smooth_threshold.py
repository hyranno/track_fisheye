import os
import wgpu
import ctypes

import wgpu_util
import texture_util


class SmoothThresholdShader(wgpu_util.RenderShader2d):
    def __init__(
        self,
        device: wgpu.GPUDevice,
        src_format: wgpu.TextureFormat,
    ):
        fs_source = open(os.path.join(os.path.dirname(__file__), 'smooth_threshold.wgsl'), "r").read()
        bind_entries = [
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "sampler": {
                    "type": wgpu.SamplerBindingType.filtering,
                }
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "texture": {
                    "sample_type": texture_util.texture_type_to_sample_type(src_format),
                    "view_dimension": wgpu.TextureViewDimension.d2,
                },
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "texture": {
                    "sample_type": texture_util.texture_type_to_sample_type(src_format),
                    "view_dimension": wgpu.TextureViewDimension.d2,
                },
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
        sampler: wgpu.GPUSampler,
        view: wgpu.GPUTextureView,
        view_mean: wgpu.GPUTextureView,
        edges: tuple[float, float],
    ):
        buffer_edges = self.device.create_buffer_with_data(
            data=(ctypes.c_float * 2)(edges[0], edges[1]),
            usage=wgpu.BufferUsage.UNIFORM
        )
        entries = [
            {"binding": 0, "resource": sampler},
            {"binding": 1, "resource": view},
            {"binding": 2, "resource": view_mean},
            {"binding": 3, "resource": {"buffer": buffer_edges, "offset": 0, "size": buffer_edges.size}},
        ]
        return super().create_bind_group(entries)
