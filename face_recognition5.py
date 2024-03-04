# face recognition part II
# IMPORT
import cv2 as cv
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet

# INITIALIZE
facenet = FaceNet()
faces_embeddings = np.load("faces_embeddings_done_4classes.npz")
Y = faces_embeddings['arr_1']
print(Y)
encoder = LabelEncoder()
encoder.fit(Y)
haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
model = pickle.load(open("svm_model_160x160.pkl", 'rb'))

cap = cv.VideoCapture(0)

# Set threshold for recognizing known faces and for considering unknown faces
threshold_known = 0.5  # Adjust this threshold for known faces
threshold_distance = 0.8  # Adjust this threshold for unknown faces

# WHILE LOOP
while cap.isOpened():
    _, frame = cap.read()
    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)

    for x, y, w, h in faces:
        img = rgb_img[y:y + h, x:x + w]
        img = cv.resize(img, (160, 160))  # 1x160x160x3
        img = np.expand_dims(img, axis=0)
        ypred = facenet.embeddings(img)

        # Calculate distances between embeddings
        distances = []
        for emb in faces_embeddings['arr_0']:
            distances.append(np.linalg.norm(ypred - emb))
            # print(distances)
        min_distance = min(distances)
        # print(len(distances))
        if min_distance < threshold_known:
            idx = distances.index(min_distance)
            print(Y[idx])
        else:
            # Check if the minimum distance is greater than the threshold for unknown faces
            if min_distance > threshold_distance:
                final_name = "Unknown"
            else:
                idx = distances.index(min_distance)
                final_name = Y[idx]

        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 10)
        cv.putText(frame, str(final_name), (x, y - 10), cv.FONT_HERSHEY_SIMPLEX,
                   1, (0, 0, 255), 3, cv.LINE_AA)

    cv.imshow("Face Recognition:", frame)
    if cv.waitKey(1) & ord('q') == 27:
        break

cap.release()
cv.destroyAllWindows()
