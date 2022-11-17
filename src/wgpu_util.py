import os
import wgpu
from wgpu.gui.auto import WgpuCanvas
import wgpu.backends.rs  # noqa: F401, Select Rust backend

import typing


shader_defaults = {
    'dir': os.path.join(os.path.dirname(__file__), 'shaders'),
    'header': open(os.path.join(os.path.dirname(__file__), 'shaders/header.wgsl'), "r").read(),
    'vert2d': open(os.path.join(os.path.dirname(__file__), 'shaders/vert_full2d.wgsl'), "r").read(),
}


class BufferResource:
    def __init__(
        self,
        buffer: wgpu.GPUBuffer,
        offset: int = -1,
        size: int = -1,
    ):
        self.buffer = buffer
        self.offset = max(0, offset)
        self.size = buffer.size if (size < 0) else size


def get_context(canvas: wgpu.WgpuCanvasInterface) -> tuple[wgpu.GPUDevice, wgpu.GPUCanvasContext]:
    adapter = wgpu.request_adapter(canvas=canvas, power_preference="high-performance")
    device = adapter.request_device()
    present_context = canvas.get_context()
    render_texture_format = present_context.get_preferred_format(device.adapter)
    present_context.configure(device=device, format=render_texture_format)
    return device, present_context


def create_canvas(title: str) -> tuple[wgpu.GPUDevice, wgpu.GPUCanvasContext]:
    canvas = WgpuCanvas(title=title)
    return get_context(canvas)


def create_sized_canvas(size: tuple[int, int], title: str) -> tuple[wgpu.GPUDevice, wgpu.GPUCanvasContext]:
    canvas = WgpuCanvas(size=size, title=title)
    return get_context(canvas)


def push_2drender_pass(
    encoder: wgpu.GPUCommandEncoder,
    target: wgpu.GPUTextureView,
    pipeline: wgpu.GPURenderPipeline,
    bind: wgpu.GPUBindGroup,
):
    render_pass = encoder.begin_render_pass(
        color_attachments=[
            {
                "view": target,
                "resolve_target": None,
                "clear_value": (0, 0, 0, 1),
                "load_op": wgpu.LoadOp.clear,
                "store_op": wgpu.StoreOp.store,
            }
        ],
    )
    render_pass.set_pipeline(pipeline)
    render_pass.set_bind_group(0, bind, [], 0, 0)
    render_pass.draw(4, 1, 0, 0)
    render_pass.end()


def push_compute_pass(
    encoder: wgpu.GPUCommandEncoder,
    pipeline: wgpu.GPURenderPipeline,
    bind: wgpu.GPUBindGroup,
    size: tuple[int, int, int],
):
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind, [], 0, 0)
    compute_pass.dispatch_workgroups(*size)  # x y z
    compute_pass.end()


class RenderShaderBinding:
    def __init__(
        self,
        device: wgpu.GPUDevice,
        source: str,
        bind_entries: list[dict[str, typing.Any]],
        vertex: dict[str, typing.Any],
        primitive: dict[str, typing.Any],
        fragment: dict[str, typing.Any],
    ):
        self.device = device
        self.source = source
        self.shader = device.create_shader_module(code=source)
        self.bind_group_layout = device.create_bind_group_layout(entries=bind_entries)
        self.pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[self.bind_group_layout])
        self.vertex = vertex
        self.primitive = primitive
        self.fragment = fragment
        vertex["module"] = self.shader
        fragment["module"] = self.shader

    def create_render_pipeline(
        self,
        targets: list[dict[str, typing.Any]]
    ) -> wgpu.GPURenderPipeline:
        fragment = self.fragment
        fragment["targets"] = targets
        return self.device.create_render_pipeline(
            layout=self.pipeline_layout,
            vertex=self.vertex,
            primitive=self.primitive,
            fragment=fragment,
            depth_stencil=None,
            multisample=None,
        )

    def create_bind_group(self, entries: list[dict[str, typing.Any]]) -> wgpu.GPUBindGroup:
        return self.device.create_bind_group(layout=self.bind_group_layout, entries=entries)


class RenderShader2d(RenderShaderBinding):
    def __init__(
        self,
        device: wgpu.GPUDevice,
        fs_source: str,
        bind_entries: list[dict[str, typing.Any]],
    ):
        source = shader_defaults['header'] + shader_defaults['vert2d'] + fs_source
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


class ComputeShaderBinding:
    def __init__(
        self,
        device: wgpu.GPUDevice,
        source: str,
        entry_point: str,
        bind_layout_entries: list[dict[str, typing.Any]],
    ):
        self.device = device
        self.source = source
        self.shader = device.create_shader_module(code=source)
        self.entry_point = entry_point
        self.bind_group_layout = device.create_bind_group_layout(entries=bind_layout_entries)
        self.pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[self.bind_group_layout])

    def create_compute_pipeline(self) -> wgpu.GPUComputePipeline:
        return self.device.create_compute_pipeline(
            layout=self.pipeline_layout,
            compute={
                "module": self.shader,
                "entry_point": self.entry_point
            },
        )

    def create_bind_group(self, entries: list[dict[str, typing.Any]]) -> wgpu.GPUBindGroup:
        return self.device.create_bind_group(layout=self.bind_group_layout, entries=entries)
