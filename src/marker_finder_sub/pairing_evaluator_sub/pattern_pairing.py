import os
import wgpu
import numpy

import wgpu_util
import texture_util


class PatternPairingShader(wgpu_util.RenderShader2d):
    def __init__(
        self,
        device: wgpu.GPUDevice,
        src_format: wgpu.TextureFormat,
    ):
        fs_source = open(os.path.join(os.path.dirname(__file__), 'pattern_pairing.wgsl'), "r").read()
        bind_layout_entries = [
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
                    "type": wgpu.BufferBindingType.read_only_storage,
                }
            },
            {
                "binding": 3,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "buffer": {
                    "type": wgpu.BufferBindingType.read_only_storage,
                }
            },
            {
                "binding": 4,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "buffer": {
                    "type": wgpu.BufferBindingType.read_only_storage,
                }
            },
        ]
        super().__init__(device, fs_source, bind_layout_entries)

    def create_bind_group(
        self,
        src_view: wgpu.GPUTextureView,
        src_sampler: wgpu.GPUSampler,
        kernel: numpy.ndarray,
        positions: numpy.ndarray,
        rotations: numpy.ndarray,
    ) -> wgpu.GPUBindGroup:
        buffer_kernel = self.device.create_buffer_with_data(
            data=kernel.astype(dtype=numpy.dtype('<f'), order='C'),
            usage=wgpu.BufferUsage.STORAGE
        )
        buffer_positions = self.device.create_buffer_with_data(
            data=positions.astype(dtype=numpy.dtype('<f'), order='C'),
            usage=wgpu.BufferUsage.STORAGE
        )
        buffer_rotations = self.device.create_buffer_with_data(
            data=rotations.astype(dtype=numpy.dtype('<f'), order='C'),
            usage=wgpu.BufferUsage.STORAGE
        )
        entries = [
            {"binding": 0, "resource": src_view},
            {"binding": 1, "resource": src_sampler},
            {"binding": 2, "resource": {"buffer": buffer_kernel, "offset": 0, "size": buffer_kernel.size}},
            {"binding": 3, "resource": {"buffer": buffer_positions, "offset": 0, "size": buffer_positions.size}},
            {"binding": 4, "resource": {"buffer": buffer_rotations, "offset": 0, "size": buffer_rotations.size}},
        ]

        return super().create_bind_group(entries)
