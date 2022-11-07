import sys
sys.path.append("src")

import wgpu_util
import cv_util
import texture_util

from fisheye_detector import FisheyeDetector, DetectorParams

import numpy
from wgpu.gui.auto import run

if __name__ == "__main__":
    args = sys.argv
    # init image path from args or default
    src_path = "resources/image_preproced.png"
    dest_path = "resources/image_matched.png"
    print("processing " + src_path)
    device, context = wgpu_util.create_sized_canvas((512, 512), "image io example")
    context_texture_view = context.get_current_texture()
    context_texture_format = context.get_preferred_format(device.adapter)
    texture_src = cv_util.imread_texture(src_path, device)
    dest_view = texture_util.create_buffer_texture(device, texture_src.size).create_view()

    def kernel_org() -> numpy.ndarray:
        core = [1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0]
        kernel = core + core[::-1]
        return numpy.array(kernel, dtype=numpy.dtype('<f'), order='C')
    filter = FisheyeDetector(
        device, texture_src.create_view(), dest_view, dest_view.texture.format,
        kernel_org(),
        DetectorParams(20, 200, 1.08, 0.6),
    )

    filter.draw()
    cv_util.imwrite_texture(dest_view.texture, 4, dest_path, device)
    texture_util.draw_texture_on_texture(dest_view.texture, context_texture_view, context_texture_format, device)
    run()
