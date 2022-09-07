import wgpu
from wgpu.gui.auto import WgpuCanvas
import wgpu.backends.rs  # noqa: F401, Select Rust backend

import typing


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


class RenderShaderBinding:
    device: wgpu.GPUDevice
    source: str
    shader: wgpu.GPUShaderModule
    bind_group_layout: wgpu.GPUBindGroupLayout
    pipeline_layout: wgpu.GPUPipelineLayout
    vertex: dict[str, typing.Any]
    primitive: dict[str, typing.Any]
    fragment: dict[str, typing.Any]

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

    def create_render_pipeline(self, targets: list[dict[str, typing.Any]]) -> wgpu.GPURenderPipeline:
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
