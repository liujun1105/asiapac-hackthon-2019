import boto3
import json
import os
import argparse

def upload_images(image_folder, collection_id, create_collection):
	client = boto3.client('rekognition')


	if create_collection:
		response = client.create_collection(CollectionId=collection_id)
		print(json.dumps(response), sort_keys=True, indent=4)


	if not os.path.exists(image_folder):
		print("cannot find folder/file %s" % (image_folder))
		exit(0)

	for (dir_path, dir_names, file_names) in os.walk(image_folder):
		for file in file_names:
			filename, file_extension = os.path.splitext(file)
			if file_extension in ['.jpg', '.png', '.jpeg']:
				file_path = os.path.join(dir_path, filename)
				print("uploading file %s" % file_path)
				
				with open(file_path, 'rb') as image_reader:					
					response = client.index_faces(
						CollectionId=collection_id,
						ExternalImageId=os.path.basename(os.path.normpath(dir_path)),
						Image={'Bytes': image_reader.read()}
					)	
					print(json.dumps(response), sort_keys=True, indent=4)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Uploading Images to AWS')
	parser.add_argument('--images', help='Image File Location', type=str, required=True)
	parser.add_argument('--collection_id', help='AWS CollectionId Used to Storing Images', type=str, required=True)
	parser.add_argument('--create_collection', help='Whether to create collection', type=bool, default=False, required=False)
	args = parser.parse_args()
	upload_images(args.images, args.collection_id, args.create_collection)
