from tracking import Tracker, Trackable
import cv2
import numpy as np
import time

frame_size = 416
frame_count = 0
min_confidence = 0.5  #어느 정도의 신뢰도를 가지고 있는가 0.5이상 필터링
min_directions = 10  

height = 0
width = 0

count_limit = 0
up_count = 0
down_count = 0
direction = ''

trackers = []
trackables = {}

file_name = './video/passenger_01.mp4'
output_name = './video/output_passenger_01.avi'

# Load Yolo
net = cv2.dnn.readNet("./model/yolov3.weights", "./model/yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize Tracker 
tracker = Tracker()

# initialize the video writer 
writer = None

def writeFrame(img):
    # use global variable, writer
    global writer
    if writer is None and output_name is not None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(output_name, fourcc, 24,
                (img.shape[1], img.shape[0]), True)

    if writer is not None:
        writer.write(img)  


vs = cv2.VideoCapture(file_name)
# loop over the frames from the video stream
while True:
        ret, frame = vs.read()
        
        if frame is None:
            print('### No more frame ###')
            break
        # Start time capture
        start_time = time.time()
        frame_count += 1

        (height, width) = frame.shape[:2]
        count_limit = height // 2
        
        # draw a horizontal line in the center of the frame
        cv2.line(frame, (0, count_limit), (width, count_limit), (0, 255, 255), 2)
        
        # construct a blob for YOLO model
        # 네트워크에서 이미지를 바로 사용할 수 없기때문에 먼저 이미지를 Blob으로 변환해야 한다.
        # Blob은 이미지에서 특징을 잡아내고 크기를 조정하는데 사용된다.
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (frame_size, frame_size), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
        rects = []

        confidences = []
        #  0에서 1까지의 탐지에 대한 신뢰도
        boxes = []
        # 감지된 개체를 둘러싼 사각형의 좌표
        
        for out in outs:
            for detection in out:
                scores = detection[5:] 
                class_id = np.argmax(scores)  
                confidence = scores[class_id]
                # Filter only 'person'
                if class_id == 0 and confidence > min_confidence:
                # 사람인 경우와 confidence 값이 50%이상 

                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    
                    confidences.append(float(confidence))

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)
        # NMS-> 여러개를 detect할 때 중복되는 것을 제외 
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                rects.append([x, y, x+w, y+h])
                label = '{:,.2%}'.format(confidences[i])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv2.putText(frame, label, (x + 5, y + 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
    
        # use Tracker
        objects = tracker.update(rects)
        # 기존 사람과 새로 들어온 사람간의 거리를  측정해서 거리가 가까우면 기존, 거리가 멀면 새로운 사람으로 측

        # loop over the trackable objects
        for (objectID, centroid) in objects.items():
                # check if a trackable object exists with the object ID
                trackable = trackables.get(objectID, None)

                if trackable is None:
                        trackable = Trackable(objectID, centroid)
                else:
                        y = [c[1] for c in trackable.centroids]
                        variation = centroid[1] - np.mean(y)  # 기존 결과의 평균값을 구해 예외 상황 처리 
                        trackable.centroids.append(centroid)
                        if variation < 0:
                            direction = 1  # 올라간다 
                        else: 
                            direction = 0  # 내려간다 
                        trackable.directions.append(direction)  # 오차를 줄이기 위해 
                        mean_directions = int(round(np.mean(trackable.directions)))
                        len_directions = len(trackable.directions)

                        # check to see if the object has been counted or not
                        if (not trackable.counted) and (len_directions > min_directions):
                                if direction == 1 and centroid[1] < count_limit:
                                        up_count += 1
                                        trackable.counted = True
                                elif direction == 0 and centroid[1] > count_limit:
                                        down_count += 1
                                        trackable.counted = True

                # store the trackable object in our dictionary
                trackables[objectID] = trackable
                text = "ID {}".format(objectID)
                cv2.putText(frame, text, (centroid[0] + 10, centroid[1] + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        info = [
            ("Up", up_count),
            ("Down", down_count),
        ]

        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, height - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        writeFrame(frame)
        
        # show the output frame
        cv2.imshow("Frame", frame)
        frame_time = time.time() - start_time 
        print("Frame {} time {}".format(frame_count, frame_time))
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
                break
            
vs.release()
writer.release()
cv2.destroyAllWindows()
