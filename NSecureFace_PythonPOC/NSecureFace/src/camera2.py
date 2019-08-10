import cv2
import os
import argparse


def capture_face_image(username, device):
    output_folder = os.path.join(r'../resources/face-images/', username)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_count = 0

    face_cascade = cv2.CascadeClassifier()
    face_cascade.load(r'../resources/face-model/haarcascade/haarcascade_frontalface_default.xml')

    camera = cv2.VideoCapture(device)
    cv2.namedWindow('Photo Capture Window')

    while True:
        ret, photo = camera.read()

        if not ret:
            break

        face_detections = face_cascade.detectMultiScale(photo)
        for (x, y, w, h) in face_detections:
            cv2.rectangle(photo, (x, y), (x + w, y + h), (255, 225, 0), 1)
        cv2.imshow('Show Photo', photo)

        k = cv2.waitKey(27)

        if k == 27:
            print("Escape hit, closing...")
            break
        elif k == 99:
            image_name = os.path.join(output_folder, "{}_{}.png".format(username, image_count))
            cv2.imwrite(image_name, photo)
            print("{} write!".format(image_name))

            image_count += 1

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tool for Manually Capture Face Images')
    parser.add_argument('--username', help='Windows Username', required=True)
    parser.add_argument('--device', help='Windows Username', type=int, required=True)

    args = parser.parse_args()

    capture_face_image(args.username, args.device)
