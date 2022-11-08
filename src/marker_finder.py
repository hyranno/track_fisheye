import util
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
from concurrent.futures import ThreadPoolExecutor


class MarkerFinder:
    def __init__(
        self,
        device: wgpu.GPUDevice,
        src_view: wgpu.GPUTextureView,
        marker_format: MarkerFormat,
        detector_params: DetectorParams = None,
        pairing_threshold: int = 80,
        print_laps: bool = False
    ):
        self.print_laps = print_laps
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
        laps = util.lap_time()

        command_encoder = self.device.create_command_encoder()
        self.preprocess_filter.push_passes_to(command_encoder)
        self.pattern_matcher.push_passes_to(command_encoder)
        self.device.queue.submit([command_encoder.finish()])
        laps = util.lap_time(laps)

        points = self.points_extractor.extract()
        laps = util.lap_time(laps)

        if (len(points[1]) < 2):
            return []
        pairing_values = self.pairing_evaluator.evaluate(points[0], points[1])
        laps = util.lap_time(laps)

        def track(i: int) -> TrackingInfo | None:
            marker = self.pair_finder.find(points[0][i], points[1], pairing_values[i])
            if not(marker is None):
                marker.id = self.id_reader.read(marker.position, marker.ax, marker.ay)
            return marker
        markers = []
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = [
                executor.submit(track, i)
                for i in range(len(pairing_values))
            ]
            markers = [f.result() for f in futures]
        laps = util.lap_time(laps)

        if self.print_laps:
            print(['{:.4f}'.format(v[1]) for v in laps])
        return [m for m in markers if not(m is None)]
