import wgpu_util
import texture_util
import cv_util
from quick_track_marker import QuickTrackMarker

import shader.pattern_pairing

import sys
import cv2
import numpy
import wgpu
from wgpu.gui.auto import run
from pyclustering.cluster import xmeans


def kernel_pn(kernel_half: numpy.ndarray) -> numpy.ndarray:
    kernel = numpy.concatenate((kernel_half, -kernel_half), axis=None)
    return kernel


def xmcluster(points) -> xmeans.xmeans:
    initial_center = xmeans.kmeans_plusplus_initializer(points, 1).initialize()
    xm = xmeans.xmeans(points, initial_center, ccore=True)
    xm.process()
    return xm


class MarkerPairFinder:
    def __init__(
        self,
        device: wgpu.GPUDevice,
        preproced_view: wgpu.GPUTextureView,
        matched_view: wgpu.GPUTextureView,
        kernel: numpy.ndarray,
        pairing_threshold: int = 80,
    ):
        self.device = device
        self.texture_size = matched_view.texture.size

        self.preproced_view = preproced_view
        self.matched_view = matched_view
        self.pairing_threshold = pairing_threshold
        self.kernel_pn = kernel_pn(kernel)

        self.pairing_shader = shader.pattern_pairing.PatternPairingShader(
            device, preproced_view.texture.format
        )
        self.pairing_pipeline = self.pairing_shader.create_render_pipeline([
            {
                "format": wgpu.TextureFormat.rgba8unorm,  # pairing_view.texture.format,
                "blend": texture_util.BLEND_STATE_REPLACE,
            },
        ])

        return

    def matched_to_points(self) -> tuple[numpy.ndarray, numpy.ndarray]:
        src_np = cv_util.texture_to_cvimage(self.device, self.matched_view.texture, 4)

        positives = list(zip(*numpy.where(127 < src_np[:, :, 2])))
        negatives = list(zip(*numpy.where(127 < src_np[:, :, 1])))

        points_position = numpy.array([])
        if 0 < len(positives):
            xm_positives = xmcluster(positives)
            points_position = numpy.fliplr(numpy.array(xm_positives.get_centers()))
        points_rotation = numpy.array([])
        if 0 < len(negatives):
            xm_negatives = xmcluster(negatives)
            points_rotation = numpy.fliplr(numpy.array(xm_negatives.get_centers()))

        return (points_position, points_rotation)

    def calc_pairing_value(self, points_position: numpy.ndarray, points_rotation: numpy.ndarray) -> numpy.ndarray:
        linear_sampler = self.device.create_sampler(min_filter="linear", mag_filter="linear")
        pairing_size = len(points_rotation), len(points_position), 1
        pairing_view = texture_util.create_buffer_texture(self.device, pairing_size).create_view()

        pairing_binds = self.pairing_shader.create_bind_group(
            self.preproced_view,
            linear_sampler,
            self.kernel_pn,
            numpy.array(points_position),
            numpy.array(points_rotation),
        )
        command_encoder = self.device.create_command_encoder()
        wgpu_util.push_2drender_pass(command_encoder, pairing_view, self.pairing_pipeline, pairing_binds)
        self.device.queue.submit([command_encoder.finish()])

        pairing_val = cv_util.texture_to_cvimage(self.device, pairing_view.texture, 4)[:, :, 0]
        return pairing_val

    def find(self) -> list[QuickTrackMarker]:
        markers = []

        points_position, points_rotation = self.matched_to_points()
        if ((len(points_position) < 1) or (len(points_rotation) < 2)):
            return markers

        pairing_val = self.calc_pairing_value(points_position, points_rotation)

        pairing_sorted_indices = numpy.argsort(pairing_val, axis=1)[::-1]
        for i in range(len(points_position)):
            axis0_index = pairing_sorted_indices[i][0]
            axis1_index = pairing_sorted_indices[i][1]
            axis0_val = pairing_val[i][axis0_index]
            axis1_val = pairing_val[i][axis1_index]
            if ((self.pairing_threshold < axis0_val) and (self.pairing_threshold < axis1_val)):
                marker = QuickTrackMarker(
                    points_position[i],
                    (points_rotation[axis0_index], points_rotation[axis1_index]),
                )
                markers.append(marker)
        return markers


if __name__ == "__main__":
    args = sys.argv
    # init image path from args or default
    original_path = "resources/image.png"
    preproced_path = "resources/image_preproced.png"
    matched_path = "resources/image_matched.png"
    print("processing " + original_path)
    device, context = wgpu_util.create_sized_canvas((512, 512), "image io example")
    context_texture_view = context.get_current_texture()
    context_texture_format = context.get_preferred_format(device.adapter)
    preproced_view = cv_util.imread_texture(preproced_path, device).create_view()
    matched_view = cv_util.imread_texture(matched_path, device).create_view()

    pair_finder = MarkerPairFinder(device, preproced_view, matched_view)
    markers = pair_finder.find()

    img_preview_marker = cv2.imread(original_path, -1)
    for m in markers:
        cv_util.draw_wireframe_cube(img_preview_marker, m.size, m.points2d[0], m.quat)
    texture_preview_marker = cv_util.cvimage_to_texture(img_preview_marker, device)
    texture_util.draw_texture_on_texture(texture_preview_marker, context_texture_view, context_texture_format, device)
    run()
