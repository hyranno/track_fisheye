import os
import wgpu
import numpy

import wgpu_util
from .buffers import ClusteringBuffers


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
                "binding": 1,  # src cluster data counts
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {
                    "type": wgpu.BufferBindingType.storage,
                }
            },
            {
                "binding": 2,  # src means
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {
                    "type": wgpu.BufferBindingType.storage,
                }
            },
            {
                "binding": 3,  # src variances
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {
                    "type": wgpu.BufferBindingType.storage,
                }
            },
            {
                "binding": 4,  # src BICs
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {
                    "type": wgpu.BufferBindingType.storage,
                }
            },

            {
                "binding": 5,  # dest data points
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {
                    "type": wgpu.BufferBindingType.read_only_storage,
                }
            },
            {
                "binding": 6,  # dest cluster data counts
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {
                    "type": wgpu.BufferBindingType.storage,
                }
            },
            {
                "binding": 7,  # dest means
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {
                    "type": wgpu.BufferBindingType.storage,
                }
            },
            {
                "binding": 8,  # dest variances
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {
                    "type": wgpu.BufferBindingType.storage,
                }
            },
            {
                "binding": 9,  # dest BICs
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {
                    "type": wgpu.BufferBindingType.storage,
                }
            },

            {
                "binding": 10,  # data range
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {
                    "type": wgpu.BufferBindingType.read_only_storage,
                }
            },
            {
                "binding": 11,  # cluster ids
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {
                    "type": wgpu.BufferBindingType.read_only_storage,
                }
            },

        ]
        super().__init__(device, cs_source, "main", bind_entries)

    def create_bind_group(
        self,
        src: ClusteringBuffers,
        dest: ClusteringBuffers,
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
            {"binding": 1, "resource": vars(src.counts)},
            {"binding": 2, "resource": vars(src.means)},
            {"binding": 3, "resource": vars(src.variances)},
            {"binding": 4, "resource": vars(src.BICs)},

            {"binding": 5, "resource": vars(dest.datas)},
            {"binding": 6, "resource": vars(dest.counts)},
            {"binding": 7, "resource": vars(dest.means)},
            {"binding": 8, "resource": vars(dest.variances)},
            {"binding": 9, "resource": vars(dest.BICs)},

            {"binding": 10, "resource": vars(wgpu_util.BufferResource(buffer_data_range))},
            {"binding": 11, "resource": vars(wgpu_util.BufferResource(buffer_cluster_ids))},
        ]
        return super().create_bind_group(entries)
