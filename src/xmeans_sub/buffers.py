import wgpu_util

import wgpu
import numpy
from ctypes import sizeof, c_float, c_int
from dataclasses import dataclass


@dataclass
class ClusterData:
    counts: numpy.ndarray  # list[int]
    means: numpy.ndarray  # list[tuple[float, float]]
    variances: numpy.ndarray  # list[tuple[float, float]]
    BIC: numpy.ndarray  # list[float]


class ClusteringBuffers:
    def __init__(self, device: wgpu.GPUDevice, datas: list[tuple[float, float]], max_num_clusters: int):
        self.device = device
        self.datas = wgpu_util.BufferResource(
            device.create_buffer_with_data(
                data=datas.astype(dtype=numpy.dtype('<f'), order='C').data,
                usage=wgpu.BufferUsage.STORAGE
                | wgpu.BufferUsage.COPY_SRC
                | wgpu.BufferUsage.COPY_DST
            )
        )
        self.counts = wgpu_util.BufferResource(
            device.create_buffer(
                size=max_num_clusters * sizeof(c_int),
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
            )
        )
        self.means = wgpu_util.BufferResource(
            device.create_buffer(
                size=max_num_clusters * (2 * sizeof(c_float)),
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
            )
        )
        self.variances = wgpu_util.BufferResource(
            device.create_buffer(
                size=max_num_clusters * (2 * sizeof(c_float)),
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
            )
        )
        self.BICs = wgpu_util.BufferResource(
            device.create_buffer(
                size=max_num_clusters * sizeof(c_float),
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
            )
        )

    def read_cluster(self) -> ClusterData:
        counts = numpy.frombuffer(
            self.device.queue.read_buffer(self.sub_buffer.counts.buffer),
            dtype=numpy.dtype('<i'),
        )
        means = numpy.frombuffer(
            self.device.queue.read_buffer(self.sub_buffer.means.buffer),
            dtype=numpy.dtype('<f'),
        ).reshape([-1, 2])
        variances = numpy.frombuffer(
            self.device.queue.read_buffer(self.sub_buffer.variances.buffer),
            dtype=numpy.dtype('<f'),
        ).reshape([-1, 2])
        BICs = numpy.frombuffer(
            self.device.queue.read_buffer(self.sub_buffer.BICs.buffer),
            dtype=numpy.dtype('<f'),
        )
        return ClusterData(counts, means, variances, BICs)

    def read_cluster_with_size(self) -> ClusterData:
        raw_data = self.read_cluster()
        valid_clusters = [i for i in range(len(raw_data.counts)) if (0 < raw_data.counts[i])]
        counts = [raw_data.counts[i] for i in valid_clusters]
        means = [raw_data.means[i] for i in valid_clusters]
        variances = [raw_data.variances[i] for i in valid_clusters]
        BICs = [raw_data.BICs[i] for i in valid_clusters]
        return ClusterData(counts, means, variances, BICs)
