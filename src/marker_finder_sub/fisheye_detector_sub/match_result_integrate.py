import os
import wgpu
import numpy

import wgpu_util
import texture_util


class MatchResultIntegrateShader(wgpu_util.RenderShader2d):
    def __init__(
        self,
        device: wgpu.GPUDevice,
        src_format: wgpu.TextureFormat,
    ):
        fs_source = open(os.path.join(os.path.dirname(__file__), 'match_result_integrate.wgsl'), "r").read()
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
                "texture": {
                    "sample_type": texture_util.texture_type_to_sample_type(src_format),
                    "view_dimension": wgpu.TextureViewDimension.d2,
                },
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "sampler": {
                    "type": wgpu.SamplerBindingType.filtering,
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
        view0: wgpu.GPUTextureView,
        view1: wgpu.GPUTextureView,
        sampler: wgpu.GPUSampler,
        threshold: float,
    ):
        buffer_threshold = self.device.create_buffer_with_data(
            data=numpy.array([threshold], dtype=numpy.dtype('<f'), order='C'),
            usage=wgpu.BufferUsage.UNIFORM
        )
        entries = [
            {"binding": 0, "resource": view0},
            {"binding": 1, "resource": view1},
            {"binding": 2, "resource": sampler},
            {"binding": 3, "resource": {"buffer": buffer_threshold, "offset": 0, "size": buffer_threshold.size}},
        ]
        return super().create_bind_group(entries)
