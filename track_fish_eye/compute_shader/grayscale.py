import os
import wgpu

# import mymodule
import wgpu_util
import texture_util


class GrayscaleShader(wgpu_util.ComputeShaderBinding):
    def __init__(
        self,
        device: wgpu.GPUDevice,
        src_format: wgpu.TextureFormat,
    ):
        header = open(os.path.join(os.path.dirname(__file__), 'util.wgsl'), "r").read()
        cs_source = open(os.path.join(os.path.dirname(__file__), 'grayscale.wgsl'), "r").read()
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
                "binding": 1,  # dest
                "visibility": wgpu.ShaderStage.COMPUTE,
                "storage_texture": {
                    "access": wgpu.StorageTextureAccess.write_only,
                    "format": wgpu.TextureFormat.rgba8snorm,
                    "view_dimension": wgpu.TextureViewDimension.d2,
                },
            },
        ]
        super().__init__(device, source, "main", bind_entries)

    def create_bind_group(self, src: wgpu.GPUTextureView, dest: wgpu.GPUTextureView):
        entries = [
            {"binding": 0, "resource": src},
            {"binding": 1, "resource": dest},
        ]
        return super().create_bind_group(entries)
