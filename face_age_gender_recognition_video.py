import numpy as np
import cv2
import time

face_model = './model/res10_300x300_ssd_iter_140000.caffemodel'
face_prototxt = './model/deploy.prototxt.txt'
age_model = './model/age_net.caffemodel'
age_prototxt = './model/age_deploy.prototxt'
gender_model = './model/gender_net.caffemodel'
gender_prototxt = './model/gender_deploy.prototxt'

age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male','Female']

title_name = 'Age and Gender Recognition'
min_confidence = 0.5
recognition_count = 0
elapsed_time = 0
OUTPUT_SIZE = (300, 300)

detector = cv2.dnn.readNetFromCaffe(face_prototxt, face_model)
age_detector = cv2.dnn.readNetFromCaffe(age_prototxt, age_model)
gender_detector = cv2.dnn.readNetFromCaffe(gender_prototxt, gender_model)

    
def detectAndDisplay(image):
    start_time = time.time()
    (h, w) = image.shape[:2]

    
    imageBlob = cv2.dnn.blobFromImage(image, 1.0, OUTPUT_SIZE,
        (104.0, 177.0, 123.0), swapRB=False, crop=False)
    # 1.0->scale, (104.0, 177.0, 123.0)-> min subtraction

    detector.setInput(imageBlob)
    detections = detector.forward()  # 얼굴 인식 

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > min_confidence:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                    (78.4263377603, 87.7689143744, 114.895847746),swapRB=False)
            
    
            age_detector.setInput(face_blob)
            age_predictions = age_detector.forward()
            age_index = age_predictions[0].argmax()
            # 8개의 class중 가장 높은 확률에 있는게 몇번째인가?
            age = age_list[age_index]
            age_confidence = age_predictions[0][age_index]
            
            gender_detector.setInput(face_blob)
            gender_predictions = gender_detector.forward()
            gender_index = gender_predictions[0].argmax()
            gender = gender_list[gender_index]
            gender_confidence = gender_predictions[0][gender_index]

            text = "{}: {:.2f}% {}: {:.2f}%".format(gender, gender_confidence*100, age, age_confidence*100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY),
                (0, 255, 0), 2)
            cv2.putText(image, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            print('==============================')
            print("Gender {} time {:.2f} %".format(gender, gender_confidence*100))
            print("Age {} time {:.2f} %".format(age, age_confidence*100))
            print("Age     Probability(%)")
            for i in range(len(age_list)):
                print("{}  {:.2f}%".format(age_list[i], age_predictions[0][i]*100))
                
            print("Gender  Probability(%)")
            for i in range(len(gender_list)):
                print("{}  {:.2f} %".format(gender_list[i], gender_predictions[0][i]*100))
                

                
    frame_time = time.time() - start_time
    global elapsed_time
    elapsed_time += frame_time
    print("Frame time {:.3f} seconds".format(frame_time))
    
    cv2.imshow(title_name, image)
    

vs = cv2.VideoCapture(0)
time.sleep(2.0)
if not vs.isOpened:
    print('### Error opening video ###')
    exit(0)
while True:
    ret, frame = vs.read()
    if frame is None:
        print('### No more frame ###')
        vs.release()
        break
    detectAndDisplay(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


vs.release()
cv2.destroyAllWindows()
