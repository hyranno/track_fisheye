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
                "binding": 1,  # cluster pairs
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {
                    "type": wgpu.BufferBindingType.read_only_storage,
                }
            },
            {
                "binding": 2,  # sub_mask
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {
                    "type": wgpu.BufferBindingType.read_only_storage,
                }
            },
            {
                "binding": 3,  # data points
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {
                    "type": wgpu.BufferBindingType.storage,
                }
            },
            {
                "binding": 4,  # dest assignments
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {
                    "type": wgpu.BufferBindingType.storage,
                }
            },
        ]
        super().__init__(device, cs_source, "main", bind_entries)

    def create_bind_group(
        self,
        data_ranges: wgpu_util.BufferResource,
        cluster_pairs: wgpu_util.BufferResource,
        sub_mask: wgpu_util.BufferResource,
        points: wgpu_util.BufferResource,
        assignments: wgpu_util.BufferResource,
    ):
        entries = [
            {"binding": 0, "resource": vars(data_ranges)},
            {"binding": 1, "resource": vars(cluster_pairs)},
            {"binding": 2, "resource": vars(sub_mask)},
            {"binding": 3, "resource": vars(points)},
            {"binding": 4, "resource": vars(assignments)},
        ]
        return super().create_bind_group(entries)
