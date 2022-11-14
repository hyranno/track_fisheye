import os
import wgpu
import numpy

import wgpu_util


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
                "binding": 3,  # cluster range
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {
                    "type": wgpu.BufferBindingType.read_only_storage,
                }
            },
            {
                "binding": 4,  # dest centers
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {
                    "type": wgpu.BufferBindingType.storage,
                }
            },
            {
                "binding": 5,  # dest cluster data counts
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
        dest_assignments: wgpu_util.BufferResource,
        cluster_range: tuple[int, int],
        dest_centers: wgpu_util.BufferResource,
        dest_counts: wgpu_util.BufferResource,
    ):
        buffer_data_range = self.device.create_buffer_with_data(
            data=numpy.array(data_range, dtype=numpy.dtype('<i'), order='C'),
            usage=wgpu.BufferUsage.STORAGE
        )
        buffer_cluster_range = self.device.create_buffer_with_data(
            data=numpy.array(cluster_range, dtype=numpy.dtype('<i'), order='C'),
            usage=wgpu.BufferUsage.STORAGE
        )
        entries = [
            {"binding": 0, "resource": vars(wgpu_util.BufferResource(buffer_data_range))},
            {"binding": 1, "resource": vars(points)},
            {"binding": 2, "resource": vars(dest_assignments)},
            {"binding": 3, "resource": vars(wgpu_util.BufferResource(buffer_cluster_range))},
            {"binding": 4, "resource": vars(dest_centers)},
            {"binding": 5, "resource": vars(dest_counts)},
        ]
        return super().create_bind_group(entries)
