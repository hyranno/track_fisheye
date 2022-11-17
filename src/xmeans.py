from xmeans_sub.k2means import K2Means
from xmeans_sub.aligner import Aligner
import wgpu_util

import wgpu
import numpy
from ctypes import sizeof, c_float, c_int
from dataclasses import dataclass


class ClusteringBuffers:
    def __init__(self, device: wgpu.GPUDevice, datas: list[tuple[float, float]], max_depth: int):
        max_num_clusters = 1 << max_depth
        self.datas = device.create_buffer_with_data(
            data=datas.astype(dtype=numpy.dtype('<f'), order='C').data,
            usage=wgpu.BufferUsage.STORAGE
            | wgpu.BufferUsage.COPY_SRC
            | wgpu.BufferUsage.COPY_DST
        )
        self.counts = device.create_buffer(
            size=max_num_clusters * sizeof(c_int),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
        )
        self.means = device.create_buffer(
            size=max_num_clusters * (2 * sizeof(c_float)),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
        )
        self.BICs = device.create_buffer(
            size=max_num_clusters * sizeof(c_float),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
        )


@dataclass
class K2Task:
    data_offset: int
    data_length: int
    cluster_ids: tuple[int, int]


class XMeans:
    def __init__(self, device: wgpu.GPUDevice, datas: list[tuple[float, float]], max_depth: int):
        self.main_buffer = ClusteringBuffers(device, datas, max_depth)
        self.sub_buffer = ClusteringBuffers(device, datas, max_depth)
        self.assignments_buffer = device.create_buffer(
            size=len(datas) * sizeof(c_int),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
        )

    def cluster_sub_ids(self, parent_id: int, depth: int) -> tuple[int, int]:
        id0 = parent_id
        id1 = parent_id | (1 << self.max_depth >> depth)
        return (id0, id1)

    def xmeans():
        k2tasks = [K2Task(0, len(datas), cluster_sub_ids(0, 0))]
        k2means()

    def xmeans_sub(
        data_offset: int,
        data_length: int,
        cluster_id: int,
        depth: int,
    ):
        if (data_length < 2):
            return
        ids = self.cluster_sub_ids(cluster_id, depth)
        k2means()  # id0, id1
        read sub_buffer.(counts, means, BICs)
        sub_BIC =
        if (distance(mean0, mean1) < min_distance && sub_BIC < BIC):
            align()
            copy()
            xmeans_sub(data_offset, counts[ids[0]], depth + 1)
            xmeans_sub(data_offset + counts[ids[0]], counts[ids[1]], depth + 1)
