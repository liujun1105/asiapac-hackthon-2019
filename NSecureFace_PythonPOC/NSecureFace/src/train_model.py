from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle 

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
f = open(r'../resources/face-recognizer/recognizer.pickle', 'wb')
f.write(pickle.dumps(recognizer))
f.close()

f = open(r'../resources/face-labels/le.pickle', 'wb')
f.write(pickle.dumps(le))
f.close()
