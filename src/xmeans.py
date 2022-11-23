from xmeans_sub.buffers import ClusterBuffer, Cluster
from xmeans_sub.k2means import K2Means
from xmeans_sub.aligner import Aligner
from xmeans_sub.partial_copy import PartialCopy
import wgpu_util

import math
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
        self.device = device
        self.max_depth = max_depth
        self.min_distance = min_distance
        self.data_len = len(datas)
        self.main_buffer = ClusterBuffer(device, datas, 1 << max_depth)
        self.sub_buffer = ClusterBuffer(device, datas, 1 << max_depth)
        self.assignments_buffer = wgpu_util.BufferResource(
            device.create_buffer(
                size=len(datas) * sizeof(c_int),
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
            )
        )
        self.k2m = K2Means(device)
        self.k2m_pipeline = self.k2m.create_compute_pipeline()
        self.aligner = Aligner(device)
        self.aligner_pipeline = self.aligner.create_compute_pipeline()
        self.partial_copier = PartialCopy(device)
        self.partial_copier_pipeline = self.partial_copier.create_compute_pipeline()

    def cluster_sub_ids(self, parent_id: int, depth: int) -> tuple[int, int]:
        id0 = parent_id
        id1 = parent_id | (1 << (self.max_depth - 1) >> depth)
        return (id0, id1)

    def xmeans(self) -> ClusterBuffer:
        depth = 0
        k2tasks = [K2Task(0, self.data_len, 0)]
        while (0 < len(k2tasks)) and (depth < self.max_depth):
            self.k2means_sub(
                [(t.data_offset, t.data_length) for t in k2tasks],
                [self.cluster_sub_ids(t.cluster_id, depth) for t in k2tasks]
            )
            sub_result = self.sub_buffer.read_clusters_raw()
            applied_tasks = self.apply_result(k2tasks, sub_result, depth)
            sub_tasks = list(itertools.chain.from_iterable(
                [self.get_sub_tasks(t, sub_result, depth) for t in applied_tasks]
            ))
            k2tasks = [t for t in sub_tasks if 2 <= t.data_length]
            # print(k2tasks)

            depth += 1
        return self.main_buffer

    def k2means_sub(self, data_ranges: list[tuple[int, int]], cluster_pairs: list[tuple[int, int]]):
        bind = self.k2m.create_bind_group(
            self.sub_buffer, self.assignments_buffer, data_ranges, cluster_pairs,
        )
        command_encoder = self.device.create_command_encoder()
        wgpu_util.push_compute_pass(command_encoder, self.k2m_pipeline, bind, (len(data_ranges), 1, 1))
        self.device.queue.submit([command_encoder.finish()])

    def apply_result(self, k2tasks: list[K2Task], sub_result: list[Cluster], depth: int) -> list[K2Task]:
        sub_clusters = [self.cluster_sub_ids(t.cluster_id, depth) for t in k2tasks]
        sub_dictances = [
            numpy.linalg.norm(
                numpy.array(sub_result[cids[1]].mean) - numpy.array(sub_result[cids[0]].mean)
            )
            for cids in sub_clusters
        ]
        sub_BICs = [
            self.calc_sub_BIC(sub_clusters[i], sub_result)
            for i in range(len(k2tasks))
        ]
        parents = self.main_buffer.read_clusters_raw()
        parent_BICs = [parents[t.cluster_id].BIC for t in k2tasks]
        filtered_tasks = [
            k2tasks[i]
            for i in range(len(k2tasks))
            if ((depth == 0) or (sub_BICs[i] < parent_BICs[i])) and (self.min_distance < sub_dictances[i])
        ]
        if 0 == len(filtered_tasks):
            return []

        bind_aligner = self.aligner.create_bind_group(
            [(t.data_offset, t.data_length) for t in filtered_tasks],
            self.sub_buffer.datas,
            self.assignments_buffer,
        )
        bind_copier = self.partial_copier.create_bind_group(
            self.sub_buffer, self.main_buffer,
            [(t.data_offset, t.data_length) for t in filtered_tasks],
            [self.cluster_sub_ids(t.cluster_id, depth) for t in filtered_tasks]
        )
        command_encoder = self.device.create_command_encoder()
        wgpu_util.push_compute_pass(
            command_encoder, self.aligner_pipeline, bind_aligner, (len(filtered_tasks), 1, 1)
        )
        wgpu_util.push_compute_pass(
            command_encoder, self.partial_copier_pipeline, bind_copier, (len(filtered_tasks), 1, 1)
        )
        self.device.queue.submit([command_encoder.finish()])

        return filtered_tasks

    def calc_sub_BIC(self, cids: tuple[int, int], sub: list[Cluster]) -> float:
        b = numpy.linalg.norm(
                numpy.array(sub[cids[0]].mean) - numpy.array(sub[cids[1]].mean)
            )**2 / (
            math.prod(numpy.array(sub[cids[0]].variance)) + math.prod(numpy.array(sub[cids[1]].variance))
        )
        k = 0.5 * (1 + math.erf(0.5 * b))
        q = 8  # 2 * 2 * num_dimension
        return (
            numpy.array(sub[cids[0]].BIC) + numpy.array(sub[cids[1]].BIC)
            - 2*math.log(0.5/k)
            + q * math.log((sub[cids[0]].count + sub[cids[1]].count)**2 / (
                sub[cids[0]].count * sub[cids[1]].count)
            )
        )

    def get_sub_tasks(self, task: K2Task, sub: list[Cluster], depth: int) -> tuple[K2Task]:
        cids = self.cluster_sub_ids(task.cluster_id, depth)
        return (
            K2Task(task.data_offset, sub[cids[0]].count, cids[0]),
            K2Task(task.data_offset + sub[cids[0]].count, sub[cids[1]].count, cids[1]),
        )
