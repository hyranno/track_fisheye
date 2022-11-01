import numpy
import ldpc

import util


class BlockEncoder:
    def __init__(self, P: numpy.ndarray):
        self.value_length = P.shape[0]
        self.code_length = P.shape[1] + P.shape[0]
        self.generator = numpy.concatenate((numpy.identity(P.shape[0]).astype(numpy.int8), P), axis=1)
        H = numpy.concatenate((P.T, numpy.identity(P.shape[1]).astype(numpy.int8)), axis=1)
        self.bpdec = ldpc.bp_decoder(H)
        values = range(1 << self.value_length)
        words = map(lambda v: tuple(self.encode(v)), values)
        self.decode_dict = dict(zip(words, values))

    def encode(self, val: int) -> numpy.ndarray:
        data = numpy.array(util.int_to_bits(val, self.value_length))
        word = numpy.dot(data, self.generator) & 1
        return word

    def decode(self, received: numpy.ndarray) -> int:
        word = tuple(self.bpdec.decode(received))
        return self.decode_dict[word] if (word in self.decode_dict.keys()) else -1
