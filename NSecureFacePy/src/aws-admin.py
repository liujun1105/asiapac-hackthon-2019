import boto3
import json


def list_faces(collection_id_):
	client = boto3.client('rekognition')
	response = client.list_faces(
		CollectionId=collection_id_
	)
	print(json.dumps(response, sort_keys=True, indent=4))


def list_collection(next_token="", max_results=5):
	if len(next_token) == 0:
		response = boto3.client('rekognition').list_collections(MaxResults=max_results)
	else:
		response = boto3.client('rekognition').list_collections(NextToken=next_token, MaxResults=max_results)

	print(json.dumps(response, sort_keys=True, indent=4))	


def delete_collection(collection_id):
	client = boto3.client('rekognition')
	response = client.delete_collection(CollectionId=collection_id)
	print(json.dumps(response, sort_keys=True, indent=4))	


if __name__ == '__main__':
	list_faces('liujunju-face-collection')
