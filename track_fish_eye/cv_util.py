
import cv2
import numpy

import wgpu
from wgpu.gui.auto import WgpuCanvas, run
import wgpu.backends.rs  # noqa: F401, Select Rust backend

import typing


def cvtype(mat: numpy.ndarray) -> int:
    cvdepth: dict[str, int] = {
        "uint8": 0,
        "int8": 1,
        "uint16": 2,
        "int16": 3,
        "int32": 4,
        "float32": 5,
        "float64": 6,
    }
    depth: int = cvdepth[str(mat.dtype)]
    channels: int = mat.shape[2] - 1
    return depth * channels * 8


def cvimage_to_texture(
    img: numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]],
    device: wgpu.GPUDevice
) -> wgpu.GPUTexture:
    texture_data = img.data
    texture_size: tuple[int, int, int] = img.shape[1], img.shape[0], 1
    texture = device.create_texture(
        size=texture_size,
        usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.COPY_SRC | wgpu.TextureUsage.TEXTURE_BINDING,
        dimension=wgpu.TextureDimension.d2,
        format=wgpu.TextureFormat.rgba8unorm,
        mip_level_count=1,
        sample_count=1,
    )
    texture_data_layout = {
        "offset": 0,
        "bytes_per_row": img.shape[1] * img.shape[2],  # texture_data.strides[0],
        "rows_per_image": 0,
    }
    texture_target = {
        "texture": texture,
        "mip_level": 0,
        "origin": (0, 0, 0),
    }
    device.queue.write_texture(
        texture_target,
        texture_data,
        texture_data_layout,
        texture_size,
    )
    return texture


def texture_to_cvimage(
    texture: wgpu.GPUTexture,  # 8unorm
    shape: tuple[int, int, int],
    device: wgpu.GPUDevice
) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.uint8]]:
    texture_data_layout = {
        "offset": 0,
        "bytes_per_row": shape[1] * shape[2],  # texture_data.strides[0],
        "rows_per_image": 0,
    }
    texture_target = {
        "texture": texture,
        "mip_level": 0,
        "origin": (0, 0, 0),
    }
    dest_image_data: memoryview = device.queue.read_texture(
        texture_target,
        texture_data_layout,
        texture.size,
    )
    return numpy.asarray(dest_image_data).reshape(shape)


def imread_texture(path: str, device: wgpu.GPUDevice) -> wgpu.GPUTexture:
    return cvimage_to_texture(cv2.imread(path, -1), device)


def imwrite_texture(texture: wgpu.GPUTexture, shape: tuple[int, int, int], path: str, device: wgpu.GPUDevice) -> None:
    cv2.imwrite(path, texture_to_cvimage(texture, shape, device))
