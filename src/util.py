import os
import numpy
import math
import time


project_root = os.path.dirname(__file__)


def lap_time(buffer: list[tuple[float, float]] | None = None) -> list[tuple[float, float]]:
    if buffer is None:
        return [(time.time(), 0.0)]
    t = time.time()
    return [*buffer, (t, t - buffer[-1][0])]


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


def qr_fisheye_kernel() -> list[int]:
    return [1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0]


def gaussian_kernel(size: int) -> list[float]:
    sigma: float = size / (2 * 3)
    half_size = (size - 1) / 2
    weights = [(math.exp(-(i - half_size) ** 2 / (2*sigma*sigma))) for i in range(size)]
    denominator = sum(weights)  # calc sum, not analytic integration, for approx error
    kernel = [v / denominator for v in weights]
    return kernel
