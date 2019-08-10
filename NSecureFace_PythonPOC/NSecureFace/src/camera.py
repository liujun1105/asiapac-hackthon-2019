import cv2
import os

camera = cv2.VideoCapture(1)

username = 'liusiming'
output_folder = os.path.join(r'../resources/face-images/', username)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

image_count = 0

while image_count < 10:
    ret, image_capture = camera.read()

    if not ret: 
        break

    k = cv2.waitKey(27)

    if k % 256 == 27:
        print("Escape hit, closing...")
        break
    else:
        image_name = os.path.join(output_folder, "{}_{}.png".format(username, image_count))
        cv2.imwrite(image_name, image_capture)
        print("{} write!".format(image_name))

    image_count += 1

camera.release()
cv2.destroyAllWindows()