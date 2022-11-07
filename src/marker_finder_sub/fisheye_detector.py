import wgpu_util
import texture_util

from .fisheye_detector_sub.multiscale_matcher1d import MultiScaleMatcher1d
from .fisheye_detector_sub.match_result_integrate import MatchResultIntegrateShader

import numpy
import wgpu


class DetectorParams:  # named tuple
    def __init__(
        self,
        scale_min: int = 20,
        scale_max: int = 200,
        scale_step: float = 1.1,
        match_threshold: float = 0.6,
    ):
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.scale_step = scale_step
        self.match_threshold = match_threshold


class FisheyeDetector:
    def __init__(
        self,
        device: wgpu.GPUDevice,
        src_view: wgpu.GPUTextureView,
        dest_view: wgpu.GPUTextureView,
        dest_format: wgpu.TextureFormat,
        kernel: numpy.ndarray,
        params: DetectorParams = None,
    ):
        self.device = device
        self.dest_view = dest_view
        self.texture_size = src_view.texture.size
        nearest_sampler = device.create_sampler(min_filter="nearest", mag_filter="nearest")
        if params is None:
            params = DetectorParams()

        self.match1d_view_vertical = texture_util.create_buffer_texture(
            device, self.texture_size, wgpu.TextureFormat.rgba8snorm,
        ).create_view()
        self.match1d_view_horizontal = texture_util.create_buffer_texture(
            device, self.texture_size, wgpu.TextureFormat.rgba8snorm,
        ).create_view()
        self.match1d_vertical = MultiScaleMatcher1d(
            device,
            src_view,
            self.match1d_view_vertical,
            self.match1d_view_vertical.texture.format,
            kernel,
            numpy.array([0.0, 1.0]),
            params.scale_min,
            params.scale_max,
            params.scale_step,
        )
        self.match1d_horizontal = MultiScaleMatcher1d(
            device,
            src_view,
            self.match1d_view_horizontal,
            self.match1d_view_horizontal.texture.format,
            kernel,
            numpy.array([1.0, 0.0]),
            params.scale_min,
            params.scale_max,
            params.scale_step,
        )

        self.match_result_shader = MatchResultIntegrateShader(
            device, self.match1d_view_vertical.texture.format
        )
        self.match_result_pipeline = self.match_result_shader.create_render_pipeline([
            {
                "format": dest_format,
                "blend": texture_util.BLEND_STATE_REPLACE,
            },
        ])
        self.match_result_bind = self.match_result_shader.create_bind_group(
            self.match1d_view_vertical,
            self.match1d_view_horizontal,
            nearest_sampler,
            params.match_threshold,
        )

        return

    def push_passes_to(self, encoder: wgpu.GPUCommandEncoder):
        self.match1d_vertical.push_passes_to(encoder)
        self.match1d_horizontal.push_passes_to(encoder)
        wgpu_util.push_2drender_pass(
            encoder, self.dest_view, self.match_result_pipeline, self.match_result_bind
        )

    def draw(self) -> None:
        command_encoder = self.device.create_command_encoder()
        self.push_passes_to(command_encoder)
        self.device.queue.submit([command_encoder.finish()])
