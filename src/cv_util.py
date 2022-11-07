
import cv2
import numpy
# import quaternion

import wgpu
import wgpu.backends.rs  # noqa: F401, Select Rust backend

from tracking_info import TrackingInfo


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
    img: numpy.ndarray[any, numpy.dtype[numpy.uint8]],
    device: wgpu.GPUDevice,
    texture: wgpu.GPUTexture = None,
) -> wgpu.GPUTexture:
    texture_data = numpy.ascontiguousarray(img[:, :, [2, 1, 0, 3]]).data  # bgra to rgba
    texture_size: tuple[int, int, int] = img.shape[1], img.shape[0], 1
    if texture is None:
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
    device: wgpu.GPUDevice,
    texture: wgpu.GPUTexture,  # 8unorm
    num_channels: int,
) -> numpy.ndarray[any, numpy.dtype[numpy.uint8]]:
    shape = texture.size[1], texture.size[0], num_channels
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
    array_bgra = numpy.asarray(dest_image_data).reshape(shape)[:, :, [2, 1, 0, 3]]
    return numpy.ascontiguousarray(array_bgra)  # to bgra


def imread_texture(path: str, device: wgpu.GPUDevice) -> wgpu.GPUTexture:
    return cvimage_to_texture(cv2.imread(path, -1), device)


def imwrite_texture(texture: wgpu.GPUTexture, num_channels: int, path: str, device: wgpu.GPUDevice) -> None:
    cv2.imwrite(path, texture_to_cvimage(device, texture, num_channels))


def draw_tracking_info(
    dest: numpy.ndarray,
    marker: TrackingInfo,
) -> None:
    position = numpy.array(marker.position).astype(numpy.int32)
    px = (position + numpy.array(marker.ax)).astype(numpy.int32)
    py = (position + numpy.array(marker.ay)).astype(numpy.int32)
    if not(marker.id is None):
        cv2.putText(
            dest, str(marker.id), position,
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0, 255), 2
        )
    thickness = 2
    red = (0, 0, 255, 255)
    green = (0, 255, 0, 255)
    blue = (255, 0, 0, 255)
    cv2.circle(dest, position, 3, red, thickness)
    cv2.circle(dest, px, 3, green, thickness)
    cv2.circle(dest, py, 3, green, thickness)
    pos3d = numpy.array([marker.position[0], marker.position[1], 0])
    c = marker.size
    plane_axes = [
        [c, 0, 0],
        [0, c, 0],
        [0, 0, -c],
    ]
    # axes = quaternion.rotate_vectors(marker.quat, plane_axes)
    axes = [numpy.dot(marker.dcm, v) for v in plane_axes]
    points = [(numpy.array(v) + pos3d)[0:2].astype(int) for v in axes]
    cv2.line(dest, position, points[0], red, thickness)
    cv2.line(dest, position, points[1], green, thickness)
    cv2.line(dest, position, points[2], blue, thickness)
