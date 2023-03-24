import os
import numpy as np
import cv2 as cv
import pickle
from facenet_pytorch import MTCNN
from keras_facenet import FaceNet
from PIL import ImageGrab

face_detector = MTCNN(thresholds=[0.7, 0.7, 0.8], keep_all=True, device='cpu')


def get_embedding(face_pixels, model=FaceNet()):
    face_pixels = face_pixels.astype('float32')
    samples = np.expand_dims(face_pixels, 0)
    y_hat = model.embeddings(samples)
    return y_hat[0]


with open('faces_svm.pkl', 'rb') as f:
    model = pickle.load(f)

with open('output_encoder.pkl', 'rb') as f:
    output_encoder = pickle.load(f)

cap = cv.VideoCapture(0)

while True:
    ############### Webcam ######################
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv.flip(frame, 1)

    ############# Capture screen ################
    # img = ImageGrab.grab(bbox=(0, 0, 800, 800))
    # frame = np.array(img)
    # frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    #############################################

    pixels = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    results, _ = face_detector.detect(pixels)
    if results is not None:
        for result in results:
            x1, y1, x2, y2 = list(map(int, result))
            x1 = 0 if x1 < 0 else x1
            y1 = 0 if y1 < 0 else y1
            face = pixels[y1:y2, x1:x2]
            face = cv.resize(face, (160, 160))
            face_emb = np.expand_dims(get_embedding(face), 0)
            probs = model.predict_proba(face_emb)
            prob_max = int(np.amax(probs)*100)/100
            if prob_max < 0.6:
                predict_name = ['unknown']
            else:
                y_hat = np.expand_dims(np.argmax(probs), 0)
                predict_name = output_encoder.inverse_transform(y_hat)
            if predict_name != None:
                cv.rectangle(frame, (x1, y1), (x2, y2),
                             (0, 255, 0), thickness=2)
                cv.putText(
                    frame, f'{predict_name[0]} {prob_max}', (x1, y1), cv.FONT_HERSHEY_COMPLEX, 0.75, (0, 255, 0), 2)
    cv.imshow('frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
