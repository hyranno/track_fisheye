import sys
sys.path.append("src")

import wgpu_util
import texture_util
import cv_util

from marker_format import MarkerFormat
from points_extractor import PointsExtractor
from pairing_evaluator import PairingEvaluator
from pair_finder import PairFinder
from id_reader import IdReader

import cv2

from wgpu.gui.auto import run

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
    src_view = cv_util.imread_texture(original_path, device).create_view()
    preproced_view = cv_util.imread_texture(preproced_path, device).create_view()
    matched_view = cv_util.imread_texture(matched_path, device).create_view()

    marker_format = MarkerFormat()
    pts_extractor = PointsExtractor(device, matched_view)
    pair_eval = PairingEvaluator(device, preproced_view, marker_format.get_pairing_kernel())
    pair_finder = PairFinder()
    id_reader = IdReader(device, src_view, marker_format)

    points = pts_extractor.extract()
    pairing_values = pair_eval.evaluate(points[0], points[1])
    markers = pair_finder.find(points[0], points[1], pairing_values)

    img_preview_marker = cv2.imread(original_path, -1)
    for m in markers:
        m.id = id_reader.read(m.position, m.ax, m.ay)
        cv_util.draw_tracking_info(img_preview_marker, m)
    texture_preview_marker = cv_util.cvimage_to_texture(img_preview_marker, device)
    texture_util.draw_texture_on_texture(texture_preview_marker, context_texture_view, context_texture_format, device)
    run()
