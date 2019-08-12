from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle 
import os
print("loading face embeddings...")
data = pickle.loads(open(r'../resources/face-training-data/embeddings.pickle', 'rb').read())

print('encoding labels ...')
le = LabelEncoder()
labels = le.fit_transform(data['names'])

print("training model ...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data['embeddings'], labels)
print(labels)

# write the actual face recognition model to disk
face_recognizer_location = r'../resources/face-recognizer'
if not os.path.exists(face_recognizer_location):
    os.makedirs(face_recognizer_location)
f = open(os.path.join(face_recognizer_location, 'recognizer.pickle'), 'wb')
f.write(pickle.dumps(recognizer))
f.close()

face_label_location = r'../resources/face-labels'
if not os.path.exists(face_label_location):
    os.makedirs(face_label_location)
f = open(os.path.join(face_label_location, 'le.pickle'), 'wb')
f.write(pickle.dumps(le))
f.close()
