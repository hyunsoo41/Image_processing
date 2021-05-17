import numpy as np
import dlib  # landmark를 인식하는데 사용 
import cv2  

# 68개의 점을 구현 
RIGHT_EYE = list(range(36, 42))  # 36~41
LEFT_EYE = list(range(42, 48))
MOUTH = list(range(48, 68))
NOSE = list(range(27, 36))
EYEBROWS = list(range(17, 27))
JAWLINE = list(range(1, 17))
ALL = list(range(0, 68))
EYES = list(range(36, 48))

predictor_file = 'shape_predictor_68_face_landmarks.dat'  # 학습된 모델 사용
image_file = 'marathon_03.jpg'

detector = dlib.get_frontal_face_detector()  
predictor = dlib.shape_predictor(predictor_file)
# 얼굴 부분을 alignment할 때 사용 

image = cv2.imread(image_file)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 인식률을 높이기 위해 rgb 3개의 채널을 1개의 채널로 단순

rects = detector(gray, 1)
# detection을 하기 전에 layer를 upscaling(detection하기 전에 layer를 몇번 적용?)
print("Number of faces detected: {}".format(len(rects)))


for (i, rect) in enumerate(rects):
    points = np.matrix([[p.x, p.y] for p in predictor(gray, rect).parts()])
    # predictor을 이용해 얼굴 part는 찾음. 2차원 배열에 점들의 x,y좌표
    show_parts = points[ALL]  
    
    for (i, point) in enumerate(show_parts):
        x = point[0,0]
        y = point[0,1]
        cv2.circle(image, (x, y), 1, (0, 255, 255), -1)
        cv2.putText(image, "{}".format(i + 1), (x, y - 2),
		cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

cv2.imshow("Face Landmark", image)
cv2.waitKey(0)   
