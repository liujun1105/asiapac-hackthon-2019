import boto3

client = boto3.client('rekognition')

collection_id = 'liujunju-face-collection'

response = client.list_faces(
    CollectionId=collection_id
)
print(response)

with open(r'/Users/junliu/Development/Project/asiapac-hackthon-2019/NSecureFace/data/face-images/liuyucheng/liuyucheng_18.jpeg', 'rb') as image_reader:
    data = image_reader.read()
    response = client.search_faces_by_image(
        CollectionId=collection_id,
        Image={
            'Bytes': data
        }
    )

    print(response)
