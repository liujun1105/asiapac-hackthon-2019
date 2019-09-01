import argparse
import os

import imutils
import numpy as np
import cv2
import configparser
import boto3
import dlib


def recognize(config_parser_, image):
    client = boto3.client('rekognition')

    ret, buffer = cv2.imencode('.jpg', image)

    response = client.search_faces_by_image(
        CollectionId=config_parser_['Face Recognizer']['amazon.rekognition.face.collection'],
        Image={'Bytes': buffer.tobytes()},
        MaxFaces=int(config_parser_['Face Recognizer']['amazon.rekognition.face.max_results'])
    )

    matches = response['FaceMatches']
    if len(matches) > 0:
        print(
            '# of faces matches is %d, label %s has the highest confidence level %f'
            % (len(matches), matches[0]['Face']['ExternalImageId'], matches[0]['Face']['Confidence'])
        )
    else:
        print('no matching face recognized')


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def t_landmark(config_parser_, image):
    ld_window_name = config_parser_['Face Detector']['face_detector.window.lm']

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(config_parser_['Face Detector']['dlib.model.predicator'])

    image = imutils.resize(image, width=640, height=480)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)

        # show the face number
        cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 255), 1)

    image = imutils.resize(image, width=640, height=480)
    cv2.imshow(ld_window_name, image)


def t_face_detection(config_parser_, image):
    (h, w) = image.shape[:2]

    detector = cv2.dnn.readNetFromCaffe(
        config_parser_['Face Detector']['caffe.model.deploy_file'],
        config_parser_['Face Detector']['caffe.model.model_file']
    )

    # get our blob which is our input image
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    detector.setInput(blob)
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

            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(image, "Face #{}".format(i+1), (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

            if int(config_parser_['Face Recognizer']['amazon.rekognition.face.enabled']) == 1:
                recognize(config_parser_, image)

    image = imutils.resize(image, width=640, height=480)
    cv2.imshow(config_parser_['Face Detector']['face_detector.window.main'], image)


def launch_camera_app(config_parser_):

    device = int(config_parser_['Device']['device'])
    print('launching device #%d' % device)
    camera = cv2.VideoCapture(device)
    main_window_name = config_parser_['Face Detector']['face_detector.window.main']

    cv2.namedWindow(main_window_name)
    cv2.resizeWindow(main_window_name, 640, 480)

    ld_window_name = config_parser_['Face Detector']['face_detector.window.lm']
    cv2.namedWindow(ld_window_name)
    cv2.resizeWindow(ld_window_name, 640, 480)

    while True:
        ret, image = camera.read()

        if not ret:
            break

        t_landmark(config_parser_, image.copy())
        t_face_detection(config_parser_, image)

        k = cv2.waitKey(27)

        if k == 27:
            print("Escape hit, closing...")
            break

    camera.release()
    cv2.destroyAllWindows()


def load_configuration(config_file_path):
    config_parser = configparser.ConfigParser()
    config_parser.read(config_file_path)
    print(config_parser.sections())
    return config_parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launch CameraApp')
    parser.add_argument('--config', type=str, required=True)

    args = parser.parse_args()

    cfg = load_configuration(args.config)
    launch_camera_app(cfg)
