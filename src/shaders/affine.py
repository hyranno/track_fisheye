import os
import numpy
import wgpu

import wgpu_util
import texture_util


class AffineShader(wgpu_util.RenderShader2d):
    def __init__(
        self,
        device: wgpu.GPUDevice,
        src_format: wgpu.TextureFormat,
    ):
        fs_source = open(os.path.join(os.path.dirname(__file__), 'affine.wgsl'), "r").read()
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
        ]
        super().__init__(device, fs_source, bind_entries)

    def create_bind_group(
        self,
        src_view: wgpu.GPUTextureView,
        src_sampler: wgpu.GPUSampler,
        affine: numpy.ndarray,
    ):
        affine_aligned = numpy.vstack([affine, [0, 0, 0]]).T
        buffer_affine = self.device.create_buffer_with_data(
            data=affine_aligned.astype(dtype=numpy.dtype('<f'), order='C'),
            usage=wgpu.BufferUsage.UNIFORM
        )
        entries = [
            {"binding": 0, "resource": src_view},
            {"binding": 1, "resource": src_sampler},
            {"binding": 2, "resource": {"buffer": buffer_affine, "offset": 0, "size": buffer_affine.size}},
        ]
        return super().create_bind_group(entries)
