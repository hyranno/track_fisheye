import wgpu_util
import texture_util
import cv_util
from quick_track_marker import QuickTrackMarker
from marker_detector import MarkerDetector

import asyncio
import aioconsole
import sys
import time
import cv2
import wgpu
from wgpu.gui.auto import run

cap = cv2.VideoCapture(0)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)

is_got, frame = cap.read()
# print(frame.shape)

device, context = wgpu_util.create_sized_canvas((width, height), "detected markers")
context_texture_view = context.get_current_texture()
context_texture_format = context.get_preferred_format(device.adapter)

texture_size = frame.shape[1], frame.shape[0], 1
src_view = texture_util.create_buffer_texture(device, texture_size).create_view()
preview_texture = texture_util.create_buffer_texture(device, texture_size)
aqueue: asyncio.Queue[str] = asyncio.Queue()
detector = MarkerDetector(device, src_view)

print('wait 3 sec to initialize')
time.sleep(3)


async def main_loop():
    is_got, frame = cap.read()
    while is_got and aqueue.empty():
        src = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        cv_util.cvimage_to_texture(src, device, src_view.texture)

        markers = detector.detect()
        for m in markers:
            cv_util.draw_wireframe_cube(src, m.size, m.points2d[0], m.quat)
        cv_util.cvimage_to_texture(src, device, preview_texture)
        texture_util.draw_texture_on_texture(
            preview_texture, context.get_current_texture(), context_texture_format, device
        )
        context.present()

        await asyncio.sleep(0.1)
        is_got, frame = cap.read()


async def wait_interrupt_key():
    s = await aioconsole.ainput('press enter to exit')
    aqueue.put_nowait(s)
    print('exitting...')


async def gathered_wait(li):
    await asyncio.gather(*li)


asyncio.run(gathered_wait([
    main_loop(),
    wait_interrupt_key(),
]))
