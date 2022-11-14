
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

        buffer_datas = device.create_buffer_with_data(
            data=datas.astype(dtype=numpy.dtype('<f'), order='C').data,
            usage=wgpu.BufferUsage.STORAGE
            | wgpu.BufferUsage.COPY_SRC
            | wgpu.BufferUsage.COPY_DST
        )

        buffer_assignments = device.create_buffer(
            size=len(datas)*sizeof(c_int),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
        )
        buffer_centers = device.create_buffer(
            size=4*(2 * sizeof(c_float)),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
        )
        buffer_counts = device.create_buffer(
            size=4 * sizeof(c_int),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
        )
        offset_datas = int(len(datas) / 2)
        offset_clusters = 2

        k2m = K2Means(device)
        pipeline = k2m.create_compute_pipeline()
        bind0 = k2m.create_bind_group(
            (0, offset_datas),
            wgpu_util.BufferResource(buffer_datas),
            wgpu_util.BufferResource(buffer_assignments),
            (0, offset_clusters),
            wgpu_util.BufferResource(buffer_centers),
            wgpu_util.BufferResource(buffer_counts),
        )
        bind1 = k2m.create_bind_group(
            (offset_datas, offset_datas),
            wgpu_util.BufferResource(buffer_datas),
            wgpu_util.BufferResource(buffer_assignments),
            (offset_clusters, offset_clusters),
            wgpu_util.BufferResource(buffer_centers),
            wgpu_util.BufferResource(buffer_counts),
        )

        command_encoder = device.create_command_encoder()
        wgpu_util.push_compute_pass(command_encoder, pipeline, bind0, (1, 1, 1))
        wgpu_util.push_compute_pass(command_encoder, pipeline, bind1, (1, 1, 1))
        device.queue.submit([command_encoder.finish()])

        res_assignments = numpy.frombuffer(
            device.queue.read_buffer(buffer_assignments),
            dtype=numpy.dtype('<i'),
        )
        res_centers = numpy.frombuffer(
            device.queue.read_buffer(buffer_centers),
            dtype=numpy.dtype('<f'),
        ).reshape([-1, 2])
        res_counts = numpy.frombuffer(
            device.queue.read_buffer(buffer_counts),
            dtype=numpy.dtype('<i'),
        )

        print("result")
        print(res_assignments)
        print(res_centers)
        print(res_counts)
        # assert 1 == 1

        aligner = Aligner(device)
        pipeline = aligner.create_compute_pipeline()
        bind0 = aligner.create_bind_group(
            (0, offset_datas),
            wgpu_util.BufferResource(buffer_datas),
            wgpu_util.BufferResource(buffer_assignments),
        )
        bind1 = aligner.create_bind_group(
            (offset_datas, offset_datas),
            wgpu_util.BufferResource(buffer_datas),
            wgpu_util.BufferResource(buffer_assignments),
        )

        command_encoder = device.create_command_encoder()
        wgpu_util.push_compute_pass(command_encoder, pipeline, bind0, (1, 1, 1))
        wgpu_util.push_compute_pass(command_encoder, pipeline, bind1, (1, 1, 1))
        device.queue.submit([command_encoder.finish()])

        res_assignments = numpy.frombuffer(
            device.queue.read_buffer(buffer_assignments),
            dtype=numpy.dtype('<i'),
        )
        res_centers = numpy.frombuffer(
            device.queue.read_buffer(buffer_centers),
            dtype=numpy.dtype('<f'),
        ).reshape([-1, 2])
        res_counts = numpy.frombuffer(
            device.queue.read_buffer(buffer_counts),
            dtype=numpy.dtype('<i'),
        )

        print("result")
        print(res_assignments)
        print(res_centers)
        print(res_counts)
        # assert 1 == 1


if __name__ == "__main__":
    t = TestK2Means()
    t.test_k2means()
