import mymodule
import wgpu_util
import texture_util
import cv_util

import wgpu
from wgpu.gui.auto import run

device, context = wgpu_util.create_canvas("image io example")

texture = cv_util.imread_texture("resources/image.png", device)
shape = texture.size[1], texture.size[0], 4
# cv_util.imwrite_texture(texture, shape, "image_out.png", device)

# texture_util.draw_texture_on_texture(texture, context.get_current_texture(), device)
context_texture_view = context.get_current_texture()
context_texture_format = wgpu.TextureFormat.bgra8unorm_srgb
texture_util.draw_texture_on_texture(texture, context_texture_view, context_texture_format, device)

run()
