import numpy as np
import cv2
import time
from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
import tkinter.scrolledtext as tkst

face_model = './model/res10_300x300_ssd_iter_140000.caffemodel'
face_prototxt = './model/deploy.prototxt.txt'
age_model = './model/age_net.caffemodel'
age_prototxt = './model/age_deploy.prototxt'
gender_model = './model/gender_net.caffemodel'
gender_prototxt = './model/gender_deploy.prototxt'
image_file = './image/marathon_02.jpg'

age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male','Female']

title_name = 'Age and Gender Recognition'
min_confidence = 0.5
min_likeness = 0.5
frame_count = 0
recognition_count = 0
elapsed_time = 0
OUTPUT_SIZE = (300, 300)

detector = cv2.dnn.readNetFromCaffe(face_prototxt, face_model)
age_detector = cv2.dnn.readNetFromCaffe(age_prototxt, age_model)
gender_detector = cv2.dnn.readNetFromCaffe(gender_prototxt, gender_model)

def selectFile():
    file_name =  filedialog.askopenfilename(initialdir = "./image",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    print('File name : ', file_name)
    read_image = cv2.imread(file_name)
    image = cv2.cvtColor(read_image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(image=image)
    (height, width) = read_image.shape[:2]
    fileLabel['text'] = file_name
    detectAndDisplay(read_image)
    
def detectAndDisplay(image):
    start_time = time.time()
    (h, w) = image.shape[:2]

    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, OUTPUT_SIZE), 1.0, OUTPUT_SIZE,
        (104.0, 177.0, 123.0), swapRB=False, crop=False)  # 모델이 인식하기 편하게 단순화

    detector.setInput(imageBlob)
    detections = detector.forward()

    log_ScrolledText.delete(1.0,END)  # 초기화 

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
            age = age_list[age_index]
            age_confidence = age_predictions[0][age_index]  # 확률 
            
            gender_detector.setInput(face_blob)
            gender_predictions = gender_detector.forward()
            gender_index = gender_predictions[0].argmax()
            gender = gender_list[gender_index]
            gender_confidence = gender_predictions[0][gender_index]

            text = "{}: {}".format(gender, age)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY),
                (0, 255, 0), 2)
            cv2.putText(image, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            
            log_ScrolledText.insert(END, "%10s %10s %10.2f %2s" % ('Gender : ', gender, gender_confidence*100, '%')+'\n', 'TITLE')
            log_ScrolledText.insert(END, "%10s %10s %10.2f %2s" % ('Age    : ', age, age_confidence*100, '%')+'\n\n', 'TITLE')
            log_ScrolledText.insert(END, "%15s %20s" % ('Age', 'Probability(%)')+'\n', 'HEADER')
            for i in range(len(age_list)):
                log_ScrolledText.insert(END, "%10s %15.2f" % (age_list[i], age_predictions[0][i]*100)+'\n')
                
            log_ScrolledText.insert(END, "%12s %20s" % ('Gender', 'Probability(%)')+'\n', 'HEADER')
            for i in range(len(gender_list)):
                log_ScrolledText.insert(END, "%10s %15.2f" % (gender_list[i], gender_predictions[0][i]*100)+'\n')
                

                
    frame_time = time.time() - start_time
    global elapsed_time
    elapsed_time += frame_time
    print("Frame {} time {:.3f} seconds".format(frame_count, frame_time))
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image = Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(image=image)
    detection.config(image=imgtk)
    detection.image = imgtk
    
    

main = Tk()
main.title(title_name)
main.geometry()

# load the input image and convert it from BGR to RGB
read_image = cv2.imread(image_file)
(height, width) = read_image.shape[:2]
image = cv2.cvtColor(read_image, cv2.COLOR_BGR2RGB)
image = Image.fromarray(image)
imgtk = ImageTk.PhotoImage(image=image)

label=Label(main, text=title_name)
label.config(font=("Courier", 18))
label.grid(row=0,column=0,columnspan=4)


fileLabel=Label(main, text=image_file)
fileLabel.grid(row=1,column=0,columnspan=2)
Button(main,text="File Select", height=2,command=lambda:selectFile()).grid(row=1, column=2, columnspan=2, sticky=(N, S, W, E))
detection=Label(main, image=imgtk)
detection.grid(row=2,column=0,columnspan=4)

log_ScrolledText = tkst.ScrolledText(main, height=20)
log_ScrolledText.grid(row=3,column=0,columnspan=4, sticky=(N, S, W, E))

log_ScrolledText.configure(font='TkFixedFont')

log_ScrolledText.tag_config('HEADER', foreground='gray', font=("Helvetica", 14))
log_ScrolledText.tag_config('TITLE', foreground='orange', font=("Helvetica", 18), underline=1, justify='center')

detectAndDisplay(read_image)

main.mainloop()
