import wgpu_util
import texture_util
import cv_util
from marker_detector import MarkerDetector

import sys
import cv2
import numpy
import wgpu
from wgpu.gui.auto import run


if __name__ == "__main__":
    args = sys.argv
    # init image path from args or default
    path = "resources/image.png"
    path_preproced = "resources/image_preproced.png"
    print("processing " + path)
    # half_kernel =

    device, context = wgpu_util.create_sized_canvas((512, 512), "marker detector example")
    context_texture_view = context.get_current_texture()
    context_texture_format = context.get_preferred_format(device.adapter)

    src = cv2.imread(path, -1)
    src_view = cv_util.cvimage_to_texture(src, device).create_view()
    detector = MarkerDetector(device, src_view)

    preproced = cv2.imread(path_preproced, -1)
    preproced_view = cv_util.cvimage_to_texture(preproced, device).create_view()

    markers = detector.detect()
    for m in markers:
        cv_util.draw_tracked_marker(src, m)
        # read id
        # draw id
    texture_preview_marker = cv_util.cvimage_to_texture(src, device)
    texture_util.draw_texture_on_texture(texture_preview_marker, context_texture_view, context_texture_format, device)
    run()
