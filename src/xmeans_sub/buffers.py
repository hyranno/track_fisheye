import wgpu_util

import wgpu
import numpy
import ctypes
import io


class Cluster(ctypes.LittleEndianStructure):
    _fields_ = [
        ("count", ctypes.c_int32),
        ("_", ctypes.c_int32),
        ("mean", ctypes.c_float * 2),
        ("variance", ctypes.c_float * 2),
        ("BIC", ctypes.c_float),
        ("_", ctypes.c_int32),
    ]

    def __repr__(self):
        return "[{}, ({}, {}), ({}, {}), {}]".format(
            self.count,
            self.mean[0], self.mean[1],
            self.variance[0], self.variance[1],
            self.BIC
        )


class ClusterBuffer:
    def __init__(self, device: wgpu.GPUDevice, datas: list[tuple[float, float]], max_num_clusters: int):
        self.device = device
        self.max_num_clusters = max_num_clusters
        self.datas = wgpu_util.BufferResource(
            device.create_buffer_with_data(
                data=datas.astype(dtype=numpy.dtype('<f'), order='C').data,
                usage=wgpu.BufferUsage.STORAGE
                | wgpu.BufferUsage.COPY_SRC
                | wgpu.BufferUsage.COPY_DST
            )
        )
        self.clusters = wgpu_util.BufferResource(
            device.create_buffer(
                size=max_num_clusters * ctypes.sizeof(Cluster),
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
            )
        )

    def read_clusters_raw(self) -> list[Cluster]:
        data = self.device.queue.read_buffer(self.clusters.buffer)
        buffer = (Cluster * self.max_num_clusters)()
        io.BytesIO(data).readinto(buffer)
        return [buffer[i] for i in range(self.max_num_clusters)]

    def read_clusters_valid_sized(self) -> list[Cluster]:
        return [v for v in self.read_clusters_raw() if (0 < v.count)]
