import wgpu_util
import texture_util
import cv_util
from marker_detector import MarkerDetector

import asyncio
import aioconsole
import time
import cv2

is_video_out_enabled = True
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

fps = 10
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
video_paths = [
    'outputs/threshold.avi',
    'outputs/matched.avi',
    'outputs/markers.avi',
]
videos = [
    cv2.VideoWriter(path, fourcc, fps, (int(width), int(height)))
    for path in video_paths
]
for v in videos:
    if not(v.isOpened()):
        print('video not is_opened')

print('wait 3 sec to initialize')
time.sleep(3)


async def main_loop():
    is_got, frame = cap.read()
    while is_got and aqueue.empty():
        src = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        cv_util.cvimage_to_texture(src, device, src_view.texture)

        try:
            markers = detector.detect()
            for m in markers:
                cv_util.draw_tracked_marker(src, m)
                # cv_util.draw_wireframe_cube(src, m.size, m.points2d[0], m.quat)
            cv_util.cvimage_to_texture(src, device, preview_texture)
            texture_util.draw_texture_on_texture(
                preview_texture, context.get_current_texture(), context_texture_format, device
            )
            context.present()
            if is_video_out_enabled:
                videos[0].write(cv2.cvtColor(
                    cv_util.texture_to_cvimage(device, detector.preproced_view.texture, 4),
                    cv2.COLOR_BGRA2BGR
                ))
                videos[1].write(cv2.cvtColor(
                    cv_util.texture_to_cvimage(device, detector.matched_view.texture, 4),
                    cv2.COLOR_BGRA2BGR
                ))
                videos[2].write(cv2.cvtColor(src, cv2.COLOR_BGRA2BGR))
        except Exception as err:
            print(err)

        await asyncio.sleep(0)
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
