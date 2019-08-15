import cv2
import os
import argparse
import numpy as np


def extract_face_images(from_dir, to_dir):

    if not os.path.exists(from_dir):
        print('can not find folder %s' % from_dir)
        exit(0)

    if not os.path.exists(to_dir):
        os.makedirs(to_dir)

    model = cv2.dnn.readNetFromCaffe(
        r'../resources/face-model/dnn/deploy.prototxt',
        r'../resources/face-model/dnn/res10_300x300_ssd_iter_140000_fp16.caffemodel'
    )

    for (dir_path, dir_names, file_names) in os.walk(from_dir):
        for file in file_names:
            # split the file name and the extension into two variables
            filename, file_extension = os.path.splitext(file)
            # check if the file extension is .png, .jpeg or .jpg
            if file_extension in ['.png', '.jpeg', '.jpg']:
                print("reading file %s" % file)
                face_image = cv2.imread(os.path.join(dir_path, file))

                (h, w) = face_image.shape[:2]

                # get our blob which is our input image
                blob = cv2.dnn.blobFromImage(cv2.resize(face_image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
                model.setInput(blob)
                face_detections = model.forward()

                if len(face_detections) > 0:
                    # we are making the assumption that each image has only ONE face,
                    # so find the bounding box with the largest probability
                    i = np.argmax(face_detections[0, 0, :, 2])
                    confidence = face_detections[0, 0, i, 2]

                    # ensure that the detection with the largest probability also
                    # means our minimum probability test (thus helping filter out weak detections)
                    if confidence > 0.5:
                        # compute the (x,y)-coordinates of the bounding box for the face
                        box = face_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        face_image = face_image[startY:endY, startX:endX]

                    image_name = os.path.join(to_dir, file)
                    cv2.imwrite(image_name, face_image)
                    print("{} write!".format(image_name))

    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tool to Extract Faces From Existing Images')
    parser.add_argument('--from-dir', help='Image Directory', required=True)
    parser.add_argument('--to-dir', help='Image Directory', required=True)

    args = parser.parse_args()

    extract_face_images(args.from_dir, args.to_dir)
