import mymodule
import wgpu_util
import texture_util
import cv_util
import quick_track_marker

import shader.pattern_pairing
import shader.draw_points

import cv2
import array
import math
import numpy
import wgpu
from wgpu.gui.auto import run
from pyclustering.cluster import xmeans


def push_2drender_pass(
    encoder: wgpu.GPUCommandEncoder,
    target: wgpu.GPUTextureView,
    pipeline: wgpu.GPURenderPipeline,
    bind: wgpu.GPUBindGroup,
):
    render_pass = encoder.begin_render_pass(
        color_attachments=[
            {
                "view": target,
                "resolve_target": None,
                "clear_value": (0, 0, 0, 1),
                "load_op": wgpu.LoadOp.clear,
                "store_op": wgpu.StoreOp.store,
            }
        ],
    )
    render_pass.set_pipeline(pipeline)
    render_pass.set_bind_group(0, bind, [], 0, 0)
    render_pass.draw(4, 1, 0, 0)
    render_pass.end()


def fisheye_kernel_ndarray() -> numpy.ndarray:
    kernel = numpy.array([
        1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0
    ], dtype=numpy.dtype('<f'), order='C')
    return kernel


def fisheye_PN_kernel_ndarray(kernel_half: numpy.ndarray) -> numpy.ndarray:
    kernel = numpy.concatenate((kernel_half, -kernel_half), axis=None)
    return kernel


def kernel1d_texture(device: wgpu.GPUDevice, kernel_ndarray: numpy.ndarray) -> wgpu.GPUTexture:
    texture_data = kernel_ndarray.data
    size = kernel_ndarray.size
    texture_size: tuple[int, int, int] = size, 1, 1
    texture = device.create_texture(
        size=texture_size,
        usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.COPY_SRC | wgpu.TextureUsage.TEXTURE_BINDING,
        dimension=wgpu.TextureDimension.d1,
        format=wgpu.TextureFormat.r32float,
        mip_level_count=1,
        sample_count=1,
    )
    texture_data_layout = {
        "offset": 0,
        "bytes_per_row": 4 * size,  # texture_data.strides[0],
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


def xmcluster(points) -> xmeans.xmeans:
    initial_center = xmeans.kmeans_plusplus_initializer(points, 1).initialize()
    xm = xmeans.xmeans(points, initial_center, ccore=True)
    xm.process()
    return xm


resolution = 512, 512
device, context = wgpu_util.create_sized_canvas(resolution, "matched points example")
context_texture_view = context.get_current_texture()
context_texture_format = context.get_preferred_format(device.adapter)

# image to points
texture_matched = cv_util.imread_texture("resources/image_matched.png", device)
view_matched = texture_matched.create_view()
shape = texture_matched.size[1], texture_matched.size[0], 4  # for numpy.ndarray

src_np = cv_util.texture_to_cvimage(texture_matched, shape, device)

threshold = 50
positives = list(zip(*numpy.where(src_np[:, :, 0] > 127+threshold)))
negatives = list(zip(*numpy.where(src_np[:, :, 0] < 127-threshold)))

xm_positives = xmcluster(positives)
xm_negatives = xmcluster(negatives)

points_position = numpy.fliplr(numpy.array(xm_positives.get_centers()))
points_rotation = numpy.fliplr(numpy.array(xm_negatives.get_centers()))

# points to marker pairs
preproced_view = cv_util.imread_texture("resources/image_preproced.png", device).create_view()
kernel = fisheye_PN_kernel_ndarray(fisheye_kernel_ndarray())

linear_sampler = device.create_sampler(min_filter="linear", mag_filter="linear")
nearest_sampler = device.create_sampler(min_filter="nearest", mag_filter="nearest")

pairing_size = len(points_rotation), len(points_position), 1
pairing_view = texture_util.create_buffer_texture(device, pairing_size).create_view()

pairing_shader = shader.pattern_pairing.PatternPairingShader(
    device, preproced_view.texture.format
)
pairing_pipeline = pairing_shader.create_render_pipeline([
    {
        "format": pairing_view.texture.format,
        "blend": texture_util.BLEND_STATE_REPLACE,
    },
])
pairing_binds = pairing_shader.create_bind_group(
    preproced_view,
    linear_sampler,
    kernel,
    numpy.array(points_position),
    numpy.array(points_rotation),
)

points_view = texture_util.create_buffer_texture(device, preproced_view.texture.size).create_view()
points_shader = shader.draw_points.DrawPointsShader(
    device, preproced_view.texture.format
)
points_pipeline = points_shader.create_render_pipeline([
    {
        "format": points_view.texture.format,
        "blend": texture_util.BLEND_STATE_REPLACE,
    },
])


def draw(device: wgpu.GPUDevice):
    command_encoder = device.create_command_encoder()

    push_2drender_pass(command_encoder, pairing_view, pairing_pipeline, pairing_binds)
    # push_2drender_pass(command_encoder, points_view, points_pipeline, pairing_binds)

    device.queue.submit([command_encoder.finish()])


draw(device)

shape = pairing_view.texture.size[1], pairing_view.texture.size[0], 4  # for numpy.ndarray
pairing_val = cv_util.texture_to_cvimage(pairing_view.texture, shape, device)[:, :, 0]
pairing_sorted_indices = numpy.argsort(pairing_val, axis=1)[::-1]

markers = []
threshold = 80
for i in range(len(points_position)):
    axis0_index = pairing_sorted_indices[i][0]
    axis1_index = pairing_sorted_indices[i][1]
    axis0_val = pairing_val[i][axis0_index]
    axis1_val = pairing_val[i][axis1_index]
    if ((threshold < axis0_val) and (threshold < axis1_val)):
        marker = quick_track_marker.QuickTrackMarker(
            points_position[i],
            (points_rotation[axis0_index], points_rotation[axis1_index]),
        )
        markers.append(marker)

# print(points_position)
# print(points_rotation)
# print(markers)

img_preview_marker = cv2.imread("resources/image.png", -1)
for m in markers:
    cv_util.draw_wireframe_cube(img_preview_marker, m.size, m.points2d[0], m.quat)
texture_preview_marker = cv_util.cvimage_to_texture(img_preview_marker, device)
texture_util.draw_texture_on_texture(texture_preview_marker, context_texture_view, context_texture_format, device)
run()
