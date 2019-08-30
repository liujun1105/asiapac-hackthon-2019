import boto3
import json

response = client.list_faces(
    CollectionId=collection_id
)
print(json.dumps(response, sort_keys=True, indent=4))

def list_collection(next_token="", max_results=5):
	response = boto3.client('rekognition').list_collections(
		next_token=next_token,
		max_results=max_results
	)
	print(json.dumps(response, sort_keys=True, indent=4))	

def delete_collection(collection_id):
	client = boto3.client('rekognition')
	response = client.delete_collection(CollectionId=collection_id)
	print(json.dumps(response, sort_keys=True, indent=4))	


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Tools to AWS')
	parser.add_argument('--delete_collection', type=str, required=False)
	parser.add_argument('--list_collection', type=str, required=False)
	parser.add_argument('--max_results')
