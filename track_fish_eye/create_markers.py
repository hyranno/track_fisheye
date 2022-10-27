import cv2
import numpy

from marker_format import MarkerFormat

path_prefix = "resources/id_markers/marker"
path_postfix = ".png"

half_kernel = [1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0]
factory = MarkerFormat(half_kernel)

for i in range(1 << 8):
    path = path_prefix + str(i) + path_postfix
    marker = factory.create_marker(i)
    print(factory.extract_id(marker))
    img = cv2.cvtColor(marker * 255, cv2.COLOR_GRAY2BGRA)
    cv2.imwrite(path, img)

print("done")
