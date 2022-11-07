import texture_util
from tracking_info import TrackingInfo
from marker_format import MarkerFormat
from marker_finder_sub.smooth_threshold import SmoothThresholdFilter
from marker_finder_sub.fisheye_detector import FisheyeDetector, DetectorParams
from marker_finder_sub.points_extractor import PointsExtractor
from marker_finder_sub.pairing_evaluator import PairingEvaluator
from marker_finder_sub.pair_finder import PairFinder
from marker_finder_sub.id_reader import IdReader

import wgpu


class MarkerFinder:
    def __init__(
        self,
        device: wgpu.GPUDevice,
        src_view: wgpu.GPUTextureView,
        marker_format: MarkerFormat,
        detector_params: DetectorParams = None,
        pairing_threshold: int = 80,
    ):
        self.device = device
        texture_size = src_view.texture.size
        self.preproced_view = texture_util.create_buffer_texture(device, texture_size).create_view()
        self.matched_view = texture_util.create_buffer_texture(device, texture_size).create_view()

        self.preprocess_filter = SmoothThresholdFilter(
            device, src_view, self.preproced_view, self.preproced_view.texture.format
        )
        self.pattern_matcher = FisheyeDetector(
            device, self.preproced_view, self.matched_view, self.matched_view.texture.format,
            marker_format.get_fisheye_kernel(), detector_params,
        )
        self.points_extractor = PointsExtractor(
            device, self.matched_view
        )
        self.pairing_evaluator = PairingEvaluator(
            device, self.preproced_view, marker_format.get_pairing_kernel()
        )
        self.pair_finder = PairFinder()
        self.id_reader = IdReader(
            device, src_view, marker_format
        )

    def find(self) -> list[TrackingInfo]:
        command_encoder = self.device.create_command_encoder()
        self.preprocess_filter.push_passes_to(command_encoder)
        self.pattern_matcher.push_passes_to(command_encoder)
        self.device.queue.submit([command_encoder.finish()])
        points = self.points_extractor.extract()
        if (len(points[1]) < 2):
            return []
        pairing_values = self.pairing_evaluator.evaluate(points[0], points[1])

        # multi threading part
        markers = self.pair_finder.find(points[0], points[1], pairing_values)
        for m in markers:
            m.id = self.id_reader.read(m.position, m.ax, m.ay)

        return markers
