import wgpu_util
import texture_util
import cv_util

from .pairing_evaluator_sub.pattern_pairing import PatternPairingShader

import numpy
import wgpu


class PairingEvaluator:
    def __init__(
        self,
        device: wgpu.GPUDevice,
        preproced_view: wgpu.GPUTextureView,
        kernel: numpy.ndarray,
    ):
        self.device = device
        self.texture_size = preproced_view.texture.size

        self.preproced_view = preproced_view
        self.kernel_pn = kernel

        self.pairing_shader = PatternPairingShader(
            device, preproced_view.texture.format
        )
        self.pairing_pipeline = self.pairing_shader.create_render_pipeline([
            {
                "format": wgpu.TextureFormat.rgba8unorm,  # pairing_view.texture.format,
                "blend": texture_util.BLEND_STATE_REPLACE,
            },
        ])

    def evaluate(self, points_position: numpy.ndarray, points_rotation: numpy.ndarray) -> numpy.ndarray:
        if ((len(points_position) < 1) or (len(points_rotation) < 2)):
            return numpy.array([])

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
