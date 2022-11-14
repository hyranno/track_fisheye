import os
import wgpu
import numpy

import wgpu_util


class Aligner(wgpu_util.ComputeShaderBinding):
    def __init__(
        self,
        device: wgpu.GPUDevice,
    ):
        cs_source = open(os.path.join(os.path.dirname(__file__), 'aligner.wgsl'), "r").read()
        bind_entries = [
            {
                "binding": 0,  # data range
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {
                    "type": wgpu.BufferBindingType.read_only_storage,
                }
            },
            {
                "binding": 1,  # data points
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {
                    "type": wgpu.BufferBindingType.storage,
                }
            },
            {
                "binding": 2,  # dest assignments
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {
                    "type": wgpu.BufferBindingType.storage,
                }
            },
        ]
        super().__init__(device, cs_source, "main", bind_entries)

    def create_bind_group(
        self,
        data_range: tuple[int, int],
        points: wgpu_util.BufferResource,
        assignments: wgpu_util.BufferResource,
    ):
        buffer_data_range = self.device.create_buffer_with_data(
            data=numpy.array(data_range, dtype=numpy.dtype('<i'), order='C'),
            usage=wgpu.BufferUsage.STORAGE
        )
        entries = [
            {"binding": 0, "resource": vars(wgpu_util.BufferResource(buffer_data_range))},
            {"binding": 1, "resource": vars(points)},
            {"binding": 2, "resource": vars(assignments)},
        ]
        return super().create_bind_group(entries)
