from xmeans_sub.buffers import ClusterBuffer, Cluster
from xmeans_sub.k2means import K2Means
from xmeans_sub.sub_mask_filter import SubMaskFilter
from xmeans_sub.aligner import Aligner
from xmeans_sub.partial_copy import PartialCopy
import wgpu_util

import wgpu
import numpy
from ctypes import sizeof, c_int
import itertools
from dataclasses import dataclass


@dataclass
class K2Task:
    data_offset: int
    data_length: int
    cluster_id: int


class XMeans:
    def __init__(
        self,
        device: wgpu.GPUDevice,
        datas: list[tuple[float, float]],
        max_depth: int,
        min_distance: float,
    ):
        max_num_clusters = 1 << max_depth
        self.device = device
        self.max_depth = max_depth
        self.min_distance = min_distance
        self.data_len = len(datas)
        self.main_buffer = ClusterBuffer(device, datas, max_num_clusters)
        self.sub_buffer = ClusterBuffer(device, datas, max_num_clusters)
        self.sub_mask_buffer = wgpu_util.BufferResource(
            device.create_buffer(
                size=max_num_clusters * sizeof(c_int),
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
            )
        )
        self.assignments_buffer = wgpu_util.BufferResource(
            device.create_buffer(
                size=len(datas) * sizeof(c_int),
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
            )
        )
        self.k2m = K2Means(device)
        self.k2m_pipeline = self.k2m.create_compute_pipeline()
        self.sub_filter = SubMaskFilter(device)
        self.sub_filter_pipeline = self.sub_filter.create_compute_pipeline()
        self.aligner = Aligner(device)
        self.aligner_pipeline = self.aligner.create_compute_pipeline()
        self.partial_copier = PartialCopy(device)
        self.partial_copier_pipeline = self.partial_copier.create_compute_pipeline()

    def cluster_sub_ids(self, parent_id: int, depth: int) -> tuple[int, int]:
        id0 = parent_id
        id1 = parent_id | (1 << (self.max_depth - 1) >> depth)
        return (id0, id1)

    def process(self) -> ClusterBuffer:
        depth = 0
        k2tasks = [K2Task(0, self.data_len, 0)]
        while (0 < len(k2tasks)) and (depth < self.max_depth):
            self.k2means(k2tasks, depth)
            k2tasks = self.generate_sub_tasks(k2tasks, depth)
            # print(k2tasks)

            depth += 1
        return self.main_buffer

    def k2means(self, tasks: list[K2Task], depth: int):
        data_ranges = [(t.data_offset, t.data_length) for t in tasks]
        cluster_pairs = [self.cluster_sub_ids(t.cluster_id, depth) for t in tasks]
        data_ranges_buffer = wgpu_util.BufferResource(self.device.create_buffer_with_data(
            data=numpy.array(data_ranges, dtype=numpy.dtype('<i'), order='C'),
            usage=wgpu.BufferUsage.STORAGE
        ))
        cluster_pairs_buffer = wgpu_util.BufferResource(self.device.create_buffer_with_data(
            data=numpy.array(cluster_pairs, dtype=numpy.dtype('<i'), order='C'),
            usage=wgpu.BufferUsage.STORAGE
        ))

        k2m_bind = self.k2m.create_bind_group(
            self.sub_buffer, self.assignments_buffer, data_ranges, cluster_pairs,
        )
        sub_filter_bind = self.sub_filter.create_bind_group(
            self.sub_buffer, self.main_buffer, cluster_pairs_buffer,
            depth, self.min_distance, self.sub_mask_buffer,
        )
        aligner_bind = self.aligner.create_bind_group(
            data_ranges_buffer, cluster_pairs_buffer, self.sub_mask_buffer,
            self.sub_buffer.datas, self.assignments_buffer,
        )
        copier_bind = self.partial_copier.create_bind_group(
            self.sub_buffer, self.main_buffer,
            data_ranges_buffer, cluster_pairs_buffer, self.sub_mask_buffer,
        )

        command_encoder = self.device.create_command_encoder()
        wgpu_util.push_compute_pass(
            command_encoder, self.k2m_pipeline, k2m_bind, (len(data_ranges), 1, 1)
        )
        wgpu_util.push_compute_pass(
            command_encoder, self.sub_filter_pipeline, sub_filter_bind, (len(data_ranges), 1, 1)
        )
        wgpu_util.push_compute_pass(
            command_encoder, self.aligner_pipeline, aligner_bind, (len(data_ranges), 1, 1)
        )
        wgpu_util.push_compute_pass(
            command_encoder, self.partial_copier_pipeline, copier_bind, (len(data_ranges), 1, 1)
        )
        self.device.queue.submit([command_encoder.finish()])

    def generate_sub_tasks(self, parents: list[K2Task], depth: int) -> list[K2Task]:
        sub_result = self.sub_buffer.read_clusters_raw()
        sub_mask = numpy.frombuffer(
            self.device.queue.read_buffer(self.sub_mask_buffer.buffer), dtype=numpy.dtype('<i')
        )
        sub_tasks = itertools.chain.from_iterable(
            [self.divide_tasks(t, sub_result, depth) for t in parents]
        )
        return [t for t in sub_tasks if (sub_mask[t.cluster_id] == 2)]

    def divide_tasks(self, task: K2Task, sub: list[Cluster], depth: int) -> tuple[K2Task]:
        cids = self.cluster_sub_ids(task.cluster_id, depth)
        return (
            K2Task(task.data_offset, sub[cids[0]].count, cids[0]),
            K2Task(task.data_offset + sub[cids[0]].count, sub[cids[1]].count, cids[1]),
        )
