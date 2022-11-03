import numpy

from block_encoder import BlockEncoder
import util


class MarkerIdEncoder:
    def __init__(self):
        value_length = 8
        code_length = 64
        mats = [numpy.identity(value_length) for i in range(value_length-1)]
        for i in range(4):
            mats[i][i][i+4] = 1
        for i in range(4, 7):
            mats[i][i+1][i-4] = 1
        mats = [numpy.fliplr(mats[i]) if (i & 1 == 0) else mats[i] for i in range(len(mats))]
        P = numpy.concatenate(mats, axis=1).astype(numpy.int8)
        self.encoder = BlockEncoder(P)
        self.value_length = value_length
        self.code_length = code_length

    def encode(self, id: int) -> numpy.ndarray:
        return self.encoder.encode(id).reshape([8, 8])

    def decode(self, code: numpy.ndarray) -> int:
        print(code)
        return self.encoder.decode(code.flatten())


class MarkerFormat:
    def __init__(
        self,
        half_kernel: list[int] = [1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0]
    ):
        quater_length = len(half_kernel)
        half_length = 2 * quater_length
        length = 4 * quater_length
        marker = numpy.zeros((length, length), dtype=numpy.int8, order='C')
        fisheye_posi = util.create_fisheye(half_kernel)
        fisheye_nega = util.create_fisheye([-v for v in half_kernel])
        marker[0:half_length, 0:half_length] = fisheye_posi
        marker[0:half_length, half_length:length] = fisheye_nega
        marker[half_length:length, 0:half_length] = fisheye_nega
        marker[half_length:length, half_length:length] = 1

        stripe = [1 if (i & 1 == 0) else -1 for i in range(quater_length)]
        marker[half_length:length, half_length:length] = util.create_fisheye(stripe)
        marker[half_length:half_length+quater_length-1, half_length:half_length+quater_length-1] = 1
        marker[half_length:half_length+quater_length-1, half_length+quater_length+1:length] = 1
        marker[half_length+quater_length+1:length, half_length:half_length+quater_length-1] = 1
        marker[half_length+quater_length+1:length, half_length+quater_length+1:length] = 1
        align_pattern = util.create_fisheye([-1, -1, 1])
        align_offset = half_length + quater_length - 3
        marker[align_offset:align_offset+6, align_offset:align_offset+6] = align_pattern

        id_half = int(quater_length - 4)
        id_length = id_half * 2
        id_offsets = (half_length + 1, half_length + quater_length + 3)
        id_blank = 0
        marker[id_offsets[0]:id_offsets[0]+id_half, id_offsets[0]:id_offsets[0]+id_half] = id_blank
        marker[id_offsets[0]:id_offsets[0]+id_half, id_offsets[1]:id_offsets[1]+id_half] = id_blank
        marker[id_offsets[1]:id_offsets[1]+id_half, id_offsets[0]:id_offsets[0]+id_half] = id_blank
        marker[id_offsets[1]:id_offsets[1]+id_half, id_offsets[1]:id_offsets[1]+id_half] = id_blank

        self.half_kernel = half_kernel
        self.marker_template = marker
        self.length = length
        self.id_length = id_length
        self.id_offsets = id_offsets
        self.id_encoder = MarkerIdEncoder()

    def create_marker(self, id: int) -> numpy.ndarray:
        code = self.id_encoder.encode(id)
        li = self.id_length
        lh = int(li / 2)
        p = self.id_offsets
        marker = numpy.vectorize(lambda v: (v + 1) / 2)(self.marker_template.copy())
        marker[p[0]:p[0]+lh, p[0]:p[0]+lh] = code[0:lh, 0:lh]
        marker[p[0]:p[0]+lh, p[1]:p[1]+lh] = code[0:lh, lh:li]
        marker[p[1]:p[1]+lh, p[0]:p[0]+lh] = code[lh:li, 0:lh]
        marker[p[1]:p[1]+lh, p[1]:p[1]+lh] = code[lh:li, lh:li]
        return marker.astype(numpy.uint8)

    def extract_id(self, marker: numpy.ndarray) -> int:
        li = self.id_length
        lh = int(li / 2)
        p = self.id_offsets
        code = numpy.zeros((li, li), dtype=numpy.int8, order='C')
        code[0:lh, 0:lh] = marker[p[0]:p[0]+lh, p[0]:p[0]+lh]
        code[0:lh, lh:li] = marker[p[0]:p[0]+lh, p[1]:p[1]+lh]
        code[lh:li, 0:lh] = marker[p[1]:p[1]+lh, p[0]:p[0]+lh]
        code[lh:li, lh:li] = marker[p[1]:p[1]+lh, p[1]:p[1]+lh]
        return self.id_encoder.decode(code)

    def get_fisheye_kernel(self) -> numpy.ndarray:
        kernel = self.half_kernel + self.half_kernel[::-1]
        return numpy.array(kernel, dtype=numpy.dtype('<f'), order='C')

    def get_pairing_kernel(self) -> numpy.ndarray:
        posi = self.half_kernel + self.half_kernel[::-1]
        nega = [-v for v in posi]
        return numpy.array(posi + nega, dtype=numpy.dtype('<f'), order='C')
