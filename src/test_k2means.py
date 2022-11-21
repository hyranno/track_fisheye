
from xmeans_sub.buffers import ClusteringBuffers
from xmeans_sub.k2means import K2Means
from xmeans_sub.aligner import Aligner
import wgpu_util

import wgpu
import numpy
from numpy.random import default_rng
from ctypes import sizeof, c_float, c_int


class TestK2Means:
    def test_k2means(self):
        c0 = numpy.array([[-0.9, -0.9], [-1.0, -1.1], [-1.1, -1.0]])
        c1 = numpy.array([[-0.9, +1.0], [-1.0, +1.1], [-1.1, +0.9]])
        c2 = numpy.array([[+1.0, -0.9], [+0.9, -1.1], [+1.1, -1.0]])
        c3 = numpy.array([[+0.9, +0.9], [+1.0, +1.1], [+1.1, +1.0]])
        datas0 = numpy.vstack([c0, c1])
        datas1 = numpy.vstack([c2, c3])
        rng = default_rng()
        rng.shuffle(datas0)
        rng.shuffle(datas1)
        datas = numpy.ascontiguousarray(
            numpy.vstack([datas0, datas1])
        ).astype(dtype=numpy.dtype('<f'), order='C')
        print(datas)

        adapter = wgpu.request_adapter(canvas=None)
        device = adapter.request_device(
            # required_limits={ 'max_compute_invocations_per_workgroup': 1024, }
        )

        num_clusters = 4
        buffers = ClusteringBuffers(device, datas, num_clusters)
        buffer_assignments = device.create_buffer(
            size=len(datas)*sizeof(c_int),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
        )

        """
        buffer_datas = device.create_buffer_with_data(
            data=datas.astype(dtype=numpy.dtype('<f'), order='C').data,
            usage=wgpu.BufferUsage.STORAGE
            | wgpu.BufferUsage.COPY_SRC
            | wgpu.BufferUsage.COPY_DST
        )
        buffer_counts = device.create_buffer(
        size=num_clusters * sizeof(c_int),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
        )
        buffer_means = device.create_buffer(
        size=num_clusters * (2 * sizeof(c_float)),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
        )
        buffer_BICs = device.create_buffer(
        size=num_clusters * sizeof(c_float),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
        )
        """
        offset_datas = int(len(datas) / 2)

        k2m = K2Means(device)
        pipeline = k2m.create_compute_pipeline()
        bind = k2m.create_bind_group(
            buffers,
            wgpu_util.BufferResource(buffer_assignments),
            [(0, offset_datas), (offset_datas, offset_datas)],
            [(0, 1), (2, 3)],
        )

        command_encoder = device.create_command_encoder()
        wgpu_util.push_compute_pass(command_encoder, pipeline, bind, (2, 1, 1))
        device.queue.submit([command_encoder.finish()])

        res_assignments = numpy.frombuffer(
            device.queue.read_buffer(buffer_assignments),
            dtype=numpy.dtype('<i'),
        )
        res_counts = numpy.frombuffer(
            device.queue.read_buffer(buffers.counts.buffer),
            dtype=numpy.dtype('<i'),
        )
        res_means = numpy.frombuffer(
            device.queue.read_buffer(buffers.means.buffer),
            dtype=numpy.dtype('<f'),
        ).reshape([-1, 2])
        res_variances = numpy.frombuffer(
            device.queue.read_buffer(buffers.variances.buffer),
            dtype=numpy.dtype('<f'),
        ).reshape([-1, 2])
        res_BICs = numpy.frombuffer(
            device.queue.read_buffer(buffers.BICs.buffer),
            dtype=numpy.dtype('<f'),
        )

        print("result")
        print(res_assignments)
        print(res_counts)
        print(res_means)
        print(res_variances)
        print(res_BICs)
        # assert 1 == 1

        aligner = Aligner(device)
        pipeline = aligner.create_compute_pipeline()
        bind = aligner.create_bind_group(
            [(0, offset_datas), (offset_datas, offset_datas)],
            buffers.datas,
            wgpu_util.BufferResource(buffer_assignments),
        )

        command_encoder = device.create_command_encoder()
        wgpu_util.push_compute_pass(command_encoder, pipeline, bind, (2, 1, 1))
        device.queue.submit([command_encoder.finish()])

        res_datas = numpy.frombuffer(
            device.queue.read_buffer(buffers.datas.buffer),
            dtype=numpy.dtype('<f'),
        ).reshape([-1, 2])
        res_assignments = numpy.frombuffer(
            device.queue.read_buffer(buffer_assignments),
            dtype=numpy.dtype('<i'),
        )

        print("result")
        print(res_datas)
        print(res_assignments)
        # assert 1 == 1


if __name__ == "__main__":
    t = TestK2Means()
    t.test_k2means()
