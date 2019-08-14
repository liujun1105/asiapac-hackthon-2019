import cv2
import os
import argparse
import numpy as np


def capture_face_image(username, device, index):
    output_folder = os.path.join(r'../resources/face-images/', username)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model = cv2.dnn.readNetFromCaffe(
        r'../resources/face-model/dnn/deploy.prototxt',
        r'../resources/face-model/dnn/weights.caffemodel'
    )

    camera = cv2.VideoCapture(device)
    cv2.namedWindow('Photo Capture Window')

    while True:
        ret, photo = camera.read()

        if not ret:
            break

        (h, w) = photo.shape[:2]

        # get our blob which is our input image
        blob = cv2.dnn.blobFromImage(cv2.resize(photo, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        model.setInput(blob)
        face_detections = model.forward()

        if len(face_detections) > 0:
            # we are making the assumption that each image has only ONE face,
            # so find the bounding box with the largest probability
            i = np.argmax(face_detections[0, 0, :, 2])
            confidence = face_detections[0, 0, i, 2]

            # ensure that the detection with the largest probability also
            # means our minimum probability test (thus helping filter out weak detections)
            if confidence > 0.7:
                # compute the (x,y)-coordinates of the bounding box for the face
                box = face_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                photo = photo[startY:endY, startX:endX]
        cv2.imshow('Show Photo', photo)

        k = cv2.waitKey(27)

        if k == 27:
            print("Escape hit, closing...")
            break
        elif k == 99:
            image_name = os.path.join(output_folder, "{}_{}.png".format(username, index))
            cv2.imwrite(image_name, photo)
            print("{} write!".format(image_name))

            index += 1

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tool for Manually Capture Face Images')
    parser.add_argument('--username', help='Windows Username', required=True)
    parser.add_argument('--device', help='Camera Index', type=int, required=True)
    parser.add_argument('--index', help='Image Index', type=int, required=True)

    args = parser.parse_args()

    capture_face_image(args.username, args.device, args.index)
