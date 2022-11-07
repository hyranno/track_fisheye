import sys
sys.path.append("src")

import wgpu_util
import cv_util
import texture_util

from smooth_threshold import SmoothThresholdFilter

from wgpu.gui.auto import run

src_path = "resources/image.png"
dest_path = "resources/image_preproced.png"
print("processing " + src_path)
device, context = wgpu_util.create_sized_canvas((512, 512), "image io example")
context_texture_view = context.get_current_texture()
context_texture_format = context.get_preferred_format(device.adapter)
texture_src = cv_util.imread_texture(src_path, device)
dest_view = texture_util.create_buffer_texture(device, texture_src.size).create_view()
filter = SmoothThresholdFilter(
    device, texture_src.create_view(), dest_view, dest_view.texture.format
)

filter.draw()
cv_util.imwrite_texture(dest_view.texture, 4, dest_path, device)
texture_util.draw_texture_on_texture(dest_view.texture, context_texture_view, context_texture_format, device)
run()
