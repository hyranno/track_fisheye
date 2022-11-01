import os
import sys
import cv2
import numpy
import wgpu
from wgpu.gui.auto import run
import ctypes

# import mymodule
import wgpu_util
import texture_util
import cv_util
from marker_detector import MarkerDetector, DetectorParams, MarkerFormat


class ImgToMarkerShader(wgpu_util.RenderShaderBinding):
    def __init__(
        self,
        device: wgpu.GPUDevice,
        src_format: wgpu.TextureFormat,
    ):
        header = open(os.path.join(os.path.dirname(__file__), 'header.wgsl'), "r").read()
        vs_source = open(os.path.join(os.path.dirname(__file__), 'vert_full2d.wgsl'), "r").read()
        fs_source = open(os.path.join(os.path.dirname(__file__), 'img_to_marker.wgsl'), "r").read()
        source = header + vs_source + fs_source

        bind_entries = [
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "texture": {
                    "sample_type": texture_util.texture_type_to_sample_type(src_format),
                    "view_dimension": wgpu.TextureViewDimension.d2,
                },
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "sampler": {
                    "type": wgpu.SamplerBindingType.filtering,
                }
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "buffer": {
                    "type": wgpu.BufferBindingType.uniform,
                }
            },
            {
                "binding": 3,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "buffer": {
                    "type": wgpu.BufferBindingType.uniform,
                }
            },
        ]

        vertex = {
            "entry_point": "vs_main",
            "buffers": [],
        }
        primitive = {
            "topology": wgpu.PrimitiveTopology.triangle_strip,
            "front_face": wgpu.FrontFace.ccw,
            "cull_mode": wgpu.CullMode.none,
        }
        fragment = {
            "entry_point": "fs_main",
        }
        super().__init__(device, source, bind_entries, vertex, primitive, fragment)

    def calc_affine(
        self,
        resolution: tuple[int, int],
        position: tuple[float, float],
        axis0: tuple[float, float],
        axis1: tuple[float, float],
    ) -> numpy.ndarray:
        p = numpy.array(position)
        a0 = axis0 - p
        a1 = axis1 - p
        d = resolution
        Am = numpy.matrix([
            [2.0, 0.0, -0.5],
            [0.0, 2.0, -0.5],
            [0.0, 0.0, 1.0],
        ])  # full size of marker
        Af = numpy.matrix([
            [a0[0], a1[0], 0.0],
            [a0[1], a1[1], 0.0],
            [0.0, 0.0, 1.0],
        ])  # rotation, skew
        Ap = numpy.matrix([
            [1.0, 0.0, p[0]],
            [0.0, 1.0, p[1]],
            [0.0, 0.0, 1.0],
        ])  # translation
        Auv = numpy.matrix([
            [1.0/d[0], 0.0, 0.0],
            [0.0, 1.0/d[1], 0.0],
            [0.0, 0.0, 1.0],
        ])  # px to uv
        return Auv @ Ap @ Af @ Am

    def create_bind_group(
        self,
        src_view: wgpu.GPUTextureView,
        src_sampler: wgpu.GPUSampler,
        position: tuple[float, float],
        axis0: tuple[float, float],
        axis1: tuple[float, float],
        marker_len: int,
    ):
        affine = self.calc_affine((src_view.texture.size[0:2]), position, axis0, axis1)
        affine_aligned = numpy.vstack([affine, [0, 0, 0]]).T
        buffer_affine = self.device.create_buffer_with_data(
            data=affine_aligned.astype(dtype=numpy.dtype('<f'), order='C'),
            usage=wgpu.BufferUsage.UNIFORM
        )
        buffer_marker_len = self.device.create_buffer_with_data(
            data=(ctypes.c_uint32)(marker_len),
            usage=wgpu.BufferUsage.UNIFORM
        )
        entries = [
            {"binding": 0, "resource": src_view},
            {"binding": 1, "resource": src_sampler},
            {"binding": 2, "resource": {"buffer": buffer_affine, "offset": 0, "size": buffer_affine.size}},
            {"binding": 3, "resource": {"buffer": buffer_marker_len, "offset": 0, "size": buffer_marker_len.size}},
        ]
        return super().create_bind_group(entries)


if __name__ == "__main__":
    args = sys.argv
    # init image path from args or default
    path = "resources/image.png"
    path_preproced = "resources/image_preproced.png"
    print("processing " + path)
    # half_kernel =
    marker_len = 128  # 9*2

    device, context = wgpu_util.create_sized_canvas((marker_len, marker_len), "marker detector example")
    context_texture_view = context.get_current_texture()
    context_texture_format = context.get_preferred_format(device.adapter)

    src = cv2.imread(path, -1)
    src_view = cv_util.cvimage_to_texture(src, device).create_view()
    detector = MarkerDetector(device, src_view, MarkerFormat(), DetectorParams(20, 200, 1.08, 0.6))

    markers = detector.detect()

    preproced = cv2.imread(path_preproced, -1)
    preproced_view = cv_util.cvimage_to_texture(preproced, device).create_view()
    marker_preview = texture_util.create_buffer_texture(device, (marker_len, marker_len, 1)).create_view()
    i2m_shader = ImgToMarkerShader(device, preproced_view.texture.format)
    i2m_pipeline = i2m_shader.create_render_pipeline([
        {
            "format": marker_preview.texture.format,
            "blend": texture_util.BLEND_STATE_REPLACE,
        },
    ])
    i2m_bind = i2m_shader.create_bind_group(
        preproced_view, device.create_sampler(min_filter="linear", mag_filter="linear"),
        markers[0].points2d[0], markers[0].points2d[1][0], markers[0].points2d[1][1],
        marker_len
    )
    command_encoder = device.create_command_encoder()
    wgpu_util.push_2drender_pass(
        command_encoder, marker_preview, i2m_pipeline, i2m_bind
    )
    device.queue.submit([command_encoder.finish()])

    texture_util.draw_texture_on_texture(marker_preview.texture, context_texture_view, context_texture_format, device)
    run()
