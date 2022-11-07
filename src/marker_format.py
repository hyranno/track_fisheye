import numpy

from block_encoder import BlockEncoder
import util


class MarkerFormat:
    def __init__(
        self,
        half_kernel: list[int] = [+1, -1, -1, +1, +1, -1, +1, -1],
        align_half_kernel: list[int] = [+1, -1, -1, +1]
    ):
        # id_length = 8
        self.id_resolution = (4, 4)
        self.code_length = 12
        P = numpy.fliplr(numpy.vstack((numpy.identity(4), numpy.identity(4)))).astype(numpy.int8)
        self.id_encoder = BlockEncoder(P)
        self.half_kernel = half_kernel
        quater_length = len(half_kernel)
        half_length = 2 * quater_length
        length = 4 * quater_length
        self.length = length

        marker = numpy.zeros((length, length), dtype=numpy.int8, order='C')

        fisheye_posi = util.create_fisheye(half_kernel)
        fisheye_nega = util.create_fisheye([-v for v in half_kernel])
        marker[0:half_length, 0:half_length] = fisheye_posi
        marker[0:half_length, half_length:length] = fisheye_nega
        marker[half_length:length, 0:half_length] = fisheye_nega

        align_pattern = util.create_fisheye(align_half_kernel)
        align_offset = half_length + quater_length - len(align_half_kernel)
        align_end = align_offset + 2 * len(align_half_kernel)
        marker[align_offset:align_end, align_offset:align_end] = align_pattern

        dummy_word = numpy.zeros(12, dtype=numpy.int8, order='C')
        self.write_word(marker, dummy_word)

        self.marker_template = marker

    def id_index_to_pos(self, i: int) -> numpy.ndarray:
        h = int((i+1)/2)
        x = min(h, 3)
        y = max(h - 3, 0)
        return numpy.array((x, y) if i & 1 == 0 else (y, x)).astype(numpy.int8)

    def write_word(self, marker: numpy.ndarray, word: numpy.ndarray):
        px_size = 4
        for i in range(self.code_length):
            p = self.id_index_to_pos(i)
            p2 = p * px_size + int(self.length / 2)
            marker[p2[0]:p2[0]+px_size, p2[1]:p2[1]+px_size] = word[i]

    def create_marker(self, id: int) -> numpy.ndarray:
        code = self.id_encoder.encode(id)
        marker = numpy.vectorize(lambda v: (v + 1) / 2)(self.marker_template.copy())
        self.write_word(marker, code)
        return marker.astype(numpy.uint8)

    def extract_id(self, id_patch: numpy.ndarray) -> int:
        code = numpy.zeros(self.code_length, dtype=numpy.int8)
        for i in range(self.code_length):
            p = self.id_index_to_pos(i)
            code[i] = id_patch[p[0], p[1]]
        return self.id_encoder.decode(code)

    def get_fisheye_kernel(self) -> numpy.ndarray:
        kernel = self.half_kernel + self.half_kernel[::-1]
        return numpy.array(kernel, dtype=numpy.dtype('<f'), order='C')

    def get_pairing_kernel(self) -> numpy.ndarray:
        posi = self.half_kernel + self.half_kernel[::-1]
        nega = [-v for v in posi]
        return numpy.array(posi + nega, dtype=numpy.dtype('<f'), order='C')

    def marker_to_id_patch(self, marker: numpy.ndarray):
        offset = int(self.length / 2)
        px_size = 4
        return marker[offset::px_size, offset::px_size]
