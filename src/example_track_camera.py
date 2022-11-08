import wgpu_util
import texture_util
import cv_util
from marker_format import MarkerFormat
from marker_finder import MarkerFinder, DetectorParams

import asyncio
import aioconsole
import time
import cv2

is_detection_time_print_enabled = False
is_video_out_enabled = False
is_canvas_out_enabled = True

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
finder = MarkerFinder(device, src_view, MarkerFormat(), DetectorParams(20, 200, 1.08, 0.6), 80, True)

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
            t = time.time()
            markers = finder.find()
            if is_detection_time_print_enabled:
                print(time.time() - t)
            for m in markers:
                cv_util.draw_tracking_info(src, m)
            if is_canvas_out_enabled:
                cv_util.cvimage_to_texture(src, device, preview_texture)
                texture_util.draw_texture_on_texture(
                    preview_texture, context.get_current_texture(), context_texture_format, device
                )
                context.present()
            if is_video_out_enabled:
                videos[0].write(cv2.cvtColor(
                    cv_util.texture_to_cvimage(device, finder.preproced_view.texture, 4),
                    cv2.COLOR_BGRA2BGR
                ))
                videos[1].write(cv2.cvtColor(
                    cv_util.texture_to_cvimage(device, finder.matched_view.texture, 4),
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
