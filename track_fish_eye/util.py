import numpy


def int_to_bits(num: int, numbits: int) -> list[int]:
    return [(num >> i) & 1 for i in range(numbits)]


def bits_to_uint(bits: list[int]) -> int:
    numbits = len(bits)
    return sum([int(bits[i]) << i for i in range(numbits)])


def fold_even(index: int, length: int) -> int:
    i2 = index % (2 * length)
    return i2 if (i2 < length) else 2 * length - 1 - i2


def create_fisheye(half_kernel: list[int]) -> numpy.ndarray:
    half_length = len(half_kernel)
    length = 2 * half_length
    pattern = numpy.zeros((length, length), dtype=numpy.int8, order='C')
    for y in range(length):
        for x in range(length):
            pattern[y][x] = half_kernel[min(fold_even(y, half_length), fold_even(x, half_length))]
    return pattern
