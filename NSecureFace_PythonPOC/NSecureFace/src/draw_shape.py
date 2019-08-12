import numpy as np
from common.colors import *
import cv2
from common.drawing import *

image = np.zeros((400, 400, 3), dtype='uint8')
image[:] = LIGHT_GRAY

cv2.line(image, (0, 0), (400, 400), GREEN, 3)
cv2.line(image, (0, 400), (400, 0), BLUE, 3)
cv2.line(image, (200, 0), (200, 400), RED, 10)
cv2.line(image, (0, 200), (400, 200), YELLOW, 10)

show_with_matplotlib(image, 'My Shape')



