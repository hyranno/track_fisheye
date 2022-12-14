import os
import wgpu
import numpy

import wgpu_util
from .buffers import ClusterBuffer


class K2Means(wgpu_util.ComputeShaderBinding):
    def __init__(
        self,
        device: wgpu.GPUDevice,
    ):
        cs_source = open(os.path.join(os.path.dirname(__file__), 'k2means.wgsl'), "r").read()
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
                    "type": wgpu.BufferBindingType.read_only_storage,
                }
            },
            {
                "binding": 2,  # dest assignments
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {
                    "type": wgpu.BufferBindingType.storage,
                }
            },
            {
                "binding": 3,  # cluster ids
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {
                    "type": wgpu.BufferBindingType.read_only_storage,
                }
            },
            {
                "binding": 4,  # dest clusters
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {
                    "type": wgpu.BufferBindingType.storage,
                }
            },
        ]
        super().__init__(device, cs_source, "main", bind_entries)

    def create_bind_group(
        self,
        buffer: ClusterBuffer,
        dest_assignments: wgpu_util.BufferResource,
        data_ranges: list[tuple[int, int]],
        cluster_ids: list[tuple[int, int]],
    ):
        buffer_data_range = self.device.create_buffer_with_data(
            data=numpy.array(data_ranges, dtype=numpy.dtype('<i'), order='C'),  # <u32
            usage=wgpu.BufferUsage.STORAGE
        )
        buffer_cluster_ids = self.device.create_buffer_with_data(
            data=numpy.array(cluster_ids, dtype=numpy.dtype('<i'), order='C'),  # u<32
            usage=wgpu.BufferUsage.STORAGE
        )
        entries = [
            {"binding": 0, "resource": vars(wgpu_util.BufferResource(buffer_data_range))},
            {"binding": 1, "resource": vars(buffer.datas)},
            {"binding": 2, "resource": vars(dest_assignments)},
            {"binding": 3, "resource": vars(wgpu_util.BufferResource(buffer_cluster_ids))},
            {"binding": 4, "resource": vars(buffer.clusters)},
        ]
        return super().create_bind_group(entries)
