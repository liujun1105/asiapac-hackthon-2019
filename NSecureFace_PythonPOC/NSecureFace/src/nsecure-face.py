import cv2
import pickle
import numpy as np

detector = cv2.dnn.readNetFromCaffee(
    r'../resources/face-model/dnn/deploy.prototxt',
    r'../resources/face-model/dnn/weights.caffemodel'
)

embedder = cv2.dnn.readNetFromTorch(r'../resources/face-trained-model/nn4.small2.v1.t7')

recognizer = pickle.loads(open(r'../resources/face-recognizer/recognizer.pickle', 'rb').read())
le = pickle.loads(open(r'../resources/face-label/le.pickle', 'rb').read())

image_counter = 0
camera = cv2.VideoCapture(1)
while True:
    ret, image = camera.read()

    if not ret:
        break

    (h, w) = image.shape[:2]

    # construct a blob from the image
    blob = cv2.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False
    )
    detector.setInput(imageBlob)
    detections = detector.forward()

    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > 0.5:
            # compute the (x,y)-coordinates of the bounding box for the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI and grab the ROI dimensions
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face with and height are sufficiently large
            if fw < 20 or fH < 20:
                continue
            # construct a blob for the face ROI, then pass the blob through our face embedding model
            # to obtain the 128-d quantification of the face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, corp=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # perform classification to recognize the face
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            # draw the bounding box of the face along with the associated probability
            text = "{}: {:2f}%".format(name, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    cv2.imshow('fr', image)
    k = cv2.waitKey(27)

    if k % 256 == 27:
        break

    image_counter += 1
camera.release()
cv2.destroyAllWindows()