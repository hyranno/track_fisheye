import os
import wgpu
import numpy

import wgpu_util
from .buffers import ClusterBuffer


class SubMaskFilter(wgpu_util.ComputeShaderBinding):
    def __init__(
        self,
        device: wgpu.GPUDevice,
    ):
        cs_source = open(os.path.join(os.path.dirname(__file__), 'sub_mask_filter.wgsl'), "r").read()
        bind_entries = [
            {
                "binding": 0,  # sub clusters
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {
                    "type": wgpu.BufferBindingType.read_only_storage,
                }
            },
            {
                "binding": 1,  # parent clusters
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {
                    "type": wgpu.BufferBindingType.read_only_storage,
                }
            },
            {
                "binding": 2,  # cluster pairs
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {
                    "type": wgpu.BufferBindingType.read_only_storage,
                }
            },
            {
                "binding": 3,  # depth
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {
                    "type": wgpu.BufferBindingType.uniform,
                }
            },
            {
                "binding": 4,  # min_distance
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {
                    "type": wgpu.BufferBindingType.uniform,
                }
            },
            {
                "binding": 5,  # sub_mask
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {
                    "type": wgpu.BufferBindingType.storage,
                }
            },
        ]
        super().__init__(device, cs_source, "main", bind_entries)

    def create_bind_group(
        self,
        subs: ClusterBuffer,
        parents: ClusterBuffer,
        cluster_pairs: wgpu_util.BufferResource,
        depth: int,
        min_distance: float,
        sub_mask: wgpu_util.BufferResource,
    ):
        buffer_depth = self.device.create_buffer_with_data(
            data=numpy.array([depth], dtype=numpy.dtype('<i'), order='C'),
            usage=wgpu.BufferUsage.UNIFORM
        )
        buffer_distance = self.device.create_buffer_with_data(
            data=numpy.array([min_distance], dtype=numpy.dtype('<f'), order='C'),
            usage=wgpu.BufferUsage.UNIFORM
        )
        entries = [
            {"binding": 0, "resource": vars(subs.clusters)},
            {"binding": 1, "resource": vars(parents.clusters)},
            {"binding": 2, "resource": vars(cluster_pairs)},
            {"binding": 3, "resource": vars(wgpu_util.BufferResource(buffer_depth))},
            {"binding": 4, "resource": vars(wgpu_util.BufferResource(buffer_distance))},
            {"binding": 5, "resource": vars(sub_mask)},
        ]
        return super().create_bind_group(entries)
