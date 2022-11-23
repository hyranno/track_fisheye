import os
import wgpu
import numpy

import wgpu_util
from .buffers import ClusterBuffer


class PartialCopy(wgpu_util.ComputeShaderBinding):
    def __init__(
        self,
        device: wgpu.GPUDevice,
    ):
        cs_source = open(os.path.join(os.path.dirname(__file__), 'partial_copy.wgsl'), "r").read()
        bind_entries = [
            {
                "binding": 0,  # src data points
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {
                    "type": wgpu.BufferBindingType.read_only_storage,
                }
            },
            {
                "binding": 1,  # src clusters
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {
                    "type": wgpu.BufferBindingType.read_only_storage,
                }
            },
            {
                "binding": 2,  # dest data points
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {
                    "type": wgpu.BufferBindingType.storage,
                }
            },
            {
                "binding": 3,  # dest clusters
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {
                    "type": wgpu.BufferBindingType.storage,
                }
            },
            {
                "binding": 4,  # data range
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {
                    "type": wgpu.BufferBindingType.read_only_storage,
                }
            },
            {
                "binding": 5,  # cluster ids
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {
                    "type": wgpu.BufferBindingType.read_only_storage,
                }
            },

        ]
        super().__init__(device, cs_source, "main", bind_entries)

    def create_bind_group(
        self,
        src: ClusterBuffer,
        dest: ClusterBuffer,
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
            {"binding": 0, "resource": vars(src.datas)},
            {"binding": 1, "resource": vars(src.clusters)},
            {"binding": 2, "resource": vars(dest.datas)},
            {"binding": 3, "resource": vars(dest.clusters)},
            {"binding": 4, "resource": vars(wgpu_util.BufferResource(buffer_data_range))},
            {"binding": 5, "resource": vars(wgpu_util.BufferResource(buffer_cluster_ids))},
        ]
        return super().create_bind_group(entries)
