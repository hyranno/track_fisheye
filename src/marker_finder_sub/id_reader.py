import wgpu

from marker_format import MarkerFormat
from .id_reader_sub.id_patch_clipper import IdPatchClipper


class IdReader:
    def __init__(
        self,
        device: wgpu.GPUDevice,
        src_view: wgpu.GPUTextureView,
        marker_format: MarkerFormat,
    ):
        self.device = device
        self.src_view = src_view
        self.marker_format = marker_format
        self.clipper = IdPatchClipper(device, src_view, marker_format.id_resolution)

    def read(
        self,
        pos: tuple[float, float],
        ax: tuple[float, float],
        ay: tuple[float, float]
    ) -> int:
        id_patch = self.clipper.clip(pos, ax, ay)
        return self.marker_format.extract_id(id_patch)
