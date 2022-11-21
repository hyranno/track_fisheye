
from xmeans import XMeans

import wgpu
import numpy
from numpy.random import default_rng


class TestXMeans:
    def test_xmeans(self):
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

        xm = XMeans(device, datas, 4, 1.0)
        buffers = xm.xmeans()

        res_assignments = numpy.frombuffer(
            device.queue.read_buffer(res.assignments_buffer),
            dtype=numpy.dtype('<i'),
        )
        res_clusters = buffers.read_cluster_with_size()

        print("result")
        print(res_assignments)
        print(res_clusters.counts)
        print(res_clusters.means)
        print(res_clusters.variances)
        print(res_clusters.BICs)
        # assert 1 == 1


if __name__ == "__main__":
    t = TestXMeans()
    t.test_xmeans()
