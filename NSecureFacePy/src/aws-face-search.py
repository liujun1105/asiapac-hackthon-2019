import boto3
import json
import cv2
import base64

client = boto3.client('rekognition')

collection_id = 'liujunju-face-collection'
device_id = 1

camera = cv2.VideoCapture(device_id)
while True:
    ret, image = camera.read()

    if not ret:
        break

    ret, buffer = cv2.imencode('.jpg', image)

    response = client.search_faces_by_image(
        CollectionId=collection_id,
        Image={'Bytes': buffer.tobytes()},
        MaxFaces=5
    )

    matches = response['FaceMatches']
    if len(matches) > 0:
        print(
            '# of faces matches is %d, label %s has the highest confidence level %f'
            % (len(matches), matches[0]['Face']['ExternalImageId'], matches[0]['Face']['Confidence'])
        )
    else:
        print('no matching face recognized')
    cv2.imshow("", image)

    k = cv2.waitKey(27)

    if k % 256 == 27:
        break

camera.release()
cv2.destroyAllWindows()
