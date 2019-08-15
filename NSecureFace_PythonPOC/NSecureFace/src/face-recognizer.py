import cv2
import pickle
import numpy as np
import argparse


def recognize(device, recognize_probability):
    detector = cv2.dnn.readNetFromCaffe(
        r'../resources/face-model/dnn/deploy.prototxt',
        r'../resources/face-model/dnn/res10_300x300_ssd_iter_140000_fp16.caffemodel'
    )

    embedder = cv2.dnn.readNetFromTorch(r'../resources/face-trained-model/nn4.small2.v1.t7')

    recognizer = pickle.loads(open(r'../resources/face-recognizer/recognizer.pickle', 'rb').read())
    label = pickle.loads(open(r'../resources/face-labels/le.pickle', 'rb').read())

    image_counter = 0
    camera = cv2.VideoCapture(device)
    while True:
        ret, image = camera.read()

        if not ret:
            break

        (h, w) = image.shape[:2]

        # construct a blob from the image
        image_blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False
        )
        detector.setInput(image_blob)
        detections = detector.forward()

        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections
            if confidence > 0.8:
                # compute the (x,y)-coordinates of the bounding box for the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # extract the face ROI and grab the ROI dimensions
                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                # ensure the face with and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue
                # construct a blob for the face ROI, then pass the blob through our face embedding model
                # to obtain the 128-d quantification of the face
                face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True)
                embedder.setInput(face_blob)
                vec = embedder.forward()

                # perform classification to recognize the face
                prediction = recognizer.predict_proba(vec)[0]
                j = np.argmax(prediction)
                probability = prediction[j]
                name = label.classes_[j]

                # Get inference time:
                t, _ = embedder.getPerfProfile()
                print(
                    'Inference time: %.2f ms, Label: %s, Probability: %2.2f%%' %
                    (t * 1000.0 / cv2.getTickFrequency(), name, probability * 100)
                )

                if probability >= recognize_probability:
                    # draw the bounding box of the face along with the associated probability
                    text = "{:.2f} ms, {}, {:2.2f}%".format(
                        (t * 1000.0 / cv2.getTickFrequency()), name, probability * 100
                    )
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
                    cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.imshow('Face Recognition', image)

        k = cv2.waitKey(27)

        if k % 256 == 27:
            break

        image_counter += 1
    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Face Recognition')
    parser.add_argument('--probability', help='Probability Threshold', default=0.8, type=float, required=True)
    parser.add_argument('--device', help='Camera Index', type=int, required=True)

    args = parser.parse_args()

    recognize(args.device, args.probability)
