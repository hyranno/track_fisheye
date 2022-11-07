import wgpu_util
import texture_util
import cv_util
from marker_format import MarkerFormat
from marker_finder import MarkerFinder, DetectorParams

import sys
import cv2
from wgpu.gui.auto import run


if __name__ == "__main__":
    args = sys.argv
    # init image path from args or default
    path = "resources/image.png"
    print("processing " + path)
    device, context = wgpu_util.create_sized_canvas((512, 512), "marker detector example")
    context_texture_view = context.get_current_texture()
    context_texture_format = context.get_preferred_format(device.adapter)

    src = cv2.imread(path, -1)
    src_view = cv_util.cvimage_to_texture(src, device).create_view()
    finder = MarkerFinder(device, src_view, MarkerFormat(), DetectorParams(20, 200, 1.08, 0.6))

    markers = finder.find()
    for m in markers:
        cv_util.draw_tracking_info(src, m)
    texture_preview_marker = cv_util.cvimage_to_texture(src, device)
    texture_util.draw_texture_on_texture(texture_preview_marker, context_texture_view, context_texture_format, device)
    run()
