import cv2
import dlib
import numpy as np

# Загрузка каскадного классификатора для распознавания лиц
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
# Список для хранения признаков лиц
features = []
# Список для хранения соответствующих имен
labels = []
#test = 0
eye_distance = 0

def get_feature(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 1:
        (x, y, w, h) = faces[0]
        face_roi = gray[y:y+h, x:x+w]
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Обнаружение глаз на лице
        eyes = eye_cascade.detectMultiScale(face_roi)
        
        if len(eyes) == 2:  # Убедитесь, что обнаружены два глаза
            eye1_x, eye1_y, eye1_w, eye1_h = eyes[0]
            eye2_x, eye2_y, eye2_w, eye2_h = eyes[1]
            
            left_eye = np.array([x + eye1_x + eye1_w // 2, y + eye1_y + eye1_h // 2])
            right_eye = np.array([x + eye2_x + eye2_w // 2, y + eye2_y + eye2_h // 2])
            
            # Расчет расстояния между глазами
            global eye_distance
            eye_distance = np.linalg.norm(left_eye - right_eye)
            return face_roi.flatten()  # Возвращение плоского массива признаков лица
        
    return None

def train(image, label):
    feature = get_feature(image)
    if feature is not None:
        features.append(feature)
        labels.append(label)

def recognize(test_image):
    test_feature = get_feature(test_image)
    global test 
    test = test_feature
    if test_feature is not None:
        print(features)

        for i in range(len(features)):
            print(test_feature)
            if np.array_equal(test_feature, features[i]):
                print(features[i])
                return labels[i]
    return "Unknown"

# Обучение модели
train(cv2.imread('face1.jpg'), "Ivan")
train(cv2.imread('face2.jpg'), "Helen")

# Распознавание
test_image = cv2.imread('face1.jpg')
recognized_label = recognize(test_image)

if recognized_label == "Unknown":
    eye_distance = 0
cv2.putText(test_image, recognized_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
cv2.putText(test_image, str(int(eye_distance)) , (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

cv2.imshow('Image', test_image)
cv2.waitKey(0)
cv2.destroyAllWindows()