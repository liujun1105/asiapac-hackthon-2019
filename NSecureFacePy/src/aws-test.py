import boto3
import json

client = boto3.client('rekognition')

collection_id = 'liujunju-face-collection'

response = client.list_faces(
    CollectionId=collection_id
)
print(json.dumps(response, sort_keys=True, indent=4))
