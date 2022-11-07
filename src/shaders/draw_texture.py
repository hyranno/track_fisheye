import os
import wgpu

import wgpu_util
import texture_util


class TextureShader(wgpu_util.RenderShader2d):
    def __init__(
        self,
        device: wgpu.GPUDevice,
        src_format: wgpu.TextureFormat,
        sampler_type: wgpu.SamplerBindingType = wgpu.SamplerBindingType.filtering,
    ):
        fs_source = open(os.path.join(os.path.dirname(__file__), 'draw_texture.wgsl'), "r").read()
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
                    "type": sampler_type,
                }
            },
        ]
        super().__init__(device, fs_source, bind_entries)

    def create_bind_group(
        self,
        src_view: wgpu.GPUTextureView,
        src_sampler: wgpu.GPUSampler,
    ):
        entries = [
            {"binding": 0, "resource": src_view},
            {"binding": 1, "resource": src_sampler},
        ]
        return super().create_bind_group(entries)
