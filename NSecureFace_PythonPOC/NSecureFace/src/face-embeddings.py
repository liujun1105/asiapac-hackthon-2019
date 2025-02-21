import cv2
import os
import numpy as np
import pickle
import imutils


model = cv2.dnn.readNetFromCaffe(
    r'../resources/face-model/dnn/deploy.prototxt',
    r'../resources/face-model/dnn/res10_300x300_ssd_iter_140000_fp16.caffemodel'
)

knownNames = []
knownEmbeddings = []

embedder = cv2.dnn.readNetFromTorch(r'../resources/face-trained-model/nn4.small2.v1.t7')

for (dir_path, dir_names, file_names) in os.walk(r'../resources/face-images/'):
    for file in file_names:
        # split the file name and the extension into two variables
        filename, file_extension = os.path.splitext(file)
        # check if the file extension is .png, .jpeg or .jpg
        if file_extension in ['.png', '.jpeg', '.jpg']:
            print("found matched file %s" % file)
            image = cv2.imread(os.path.join(dir_path, file))
            image = imutils.resize(image, width=600)
            # accessing the image.shape tuple and taking the elements
            (h, w) = image.shape[:2]

            # get our blob which is our input image
            blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            # input the blob into the model and get back detections
            model.setInput(blob)
            detections = model.forward()

            print("# detections found %d" % len(detections))
            if len(detections) > 0:
                # we are making the assumption that each image has only ONE face,
                # so find the bounding box with the largest probability
                i = np.argmax(detections[0,0, :,2])
                confidence = detections[0, 0, i, 2]

                # ensure that the detection with the largest probability also
                # means our minimum probability test (thus helping filter out weak detections)
                if confidence > 0.7:
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
                    faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True)
                    embedder.setInput(faceBlob)
                    vec = embedder.forward()
                    knownNames.append(os.path.basename(os.path.normpath(dir_path)))
                    knownEmbeddings.append(vec.flatten())

data = {"embeddings": knownEmbeddings, "names": knownNames}
print(data['names'])
f = open(r'../resources/face-training-data/embeddings.pickle', 'wb')
f.write(pickle.dumps(data))
f.close()
