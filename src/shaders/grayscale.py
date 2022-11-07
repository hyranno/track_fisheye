import os
import wgpu

import wgpu_util
import texture_util


class GrayscaleShader(wgpu_util.RenderShader2d):
    def __init__(
        self,
        device: wgpu.GPUDevice,
        src_format: wgpu.TextureFormat,
    ):
        fs_source = open(os.path.join(os.path.dirname(__file__), 'grayscale.wgsl'), "r").read()
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
        ]
        super().__init__(device, fs_source, bind_entries)

    def create_bind_group(self, view: wgpu.GPUTextureView, sampler: wgpu.GPUSampler):
        entries = [
            {"binding": 0, "resource": sampler},
            {"binding": 1, "resource": view},
        ]
        return super().create_bind_group(entries)
