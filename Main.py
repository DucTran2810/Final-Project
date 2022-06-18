import numpy as np
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import os
from tkinter import *
from PIL import Image, ImageTk
import tkinter.font as font
import mtcnn
from keras.preprocessing.image import img_to_array
from keras.models import load_model
#Load model gender
gender_model = load_model('C:/Users/admin/Desktop/All/model_gender_official.h5')
gender_labels = ['female','male']
#Load model emotion
emotion_model = load_model('C:/Users/admin/Desktop/All/model_emotion_official.h5')
emotion_labels = ['angry','disgust','fear','happy','neutral','sad','surprise']
#Load model age
age_model = load_model('C:/Users/admin/Desktop/All/model_age_official_2.h5')
age_labels = ['1-6','7-13','14-20','21-27','28-34','35-41','42-48','49-55','56-61']
#Call function detect-face MTCNN
face_detector = mtcnn.MTCNN()
 # Create a GUI window
root = Tk()
# Set the background colour of GUI window
root.configure(background = "white")
# Set the title of GUI window
root.title("Detect Age")
# Set the configuration of GUI window
root.geometry("750x770")
mainFrame = Frame(root)
mainFrame.place(x=50, y=70) 
#Capture video frames
lmain =Label(mainFrame)
lmain.grid(row=1,column=0)
lmain1 =Label(mainFrame)
lmain1.grid(row=2,column=1)
cap = cv2.VideoCapture(0)
#Write function recognize_age
def recognize_age():
    ret,frame=cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # frame_rgb = frame 
    results = face_detector.detect_faces(frame_rgb)
    for res in results:
        x1, y1, width, height = res['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # EMOTION RECOGNIZE
        cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0,255,0), thickness=4)
        roi_gray = frame_rgb[y1:y2,x1:x2]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA) 
        roi = roi_gray.astype('float')/255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi,axis=0)
        prediction = emotion_model.predict(roi)[0]
        label=emotion_labels[prediction.argmax()]
        label_position = (x1-80,y1)
        cv2.putText(frame_rgb,"Emotion:",(x1-230,y1),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
        cv2.putText(frame_rgb,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

        # GENDER RECOGNIZE
        roi_color=frame_rgb[y1:y2,x1:x2]
        roi_color=cv2.resize(roi_color,(150,150),interpolation=cv2.INTER_AREA)
        roi = roi_color.astype('float')/255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi,axis=0)
        prediction = gender_model.predict(roi)[0]
        gender_label=gender_labels[prediction.argmax()]
        gender_label_position=(x1-80,y1+30) #50 pixels below to move the label outside the face
        cv2.putText(frame_rgb,"Gender: ",(x1-230,y1+30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
        cv2.putText(frame_rgb,gender_label,gender_label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

        # AGE RECOGNIZE
        roi_color1=frame_rgb[y1:y2,x1:x2]
        roi_color1=cv2.resize(roi_color1,(150,150),interpolation=cv2.INTER_AREA)
        roi = roi_color1.astype('float')/255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi,axis=0)
        prediction = age_model.predict(roi)[0]
        age_label=age_labels[prediction.argmax()]
        age_label_position=(x1-80,y1+60) #50 pixels below to move the label outside the face
        cv2.putText(frame_rgb,"Age: ",(x1-230,y1+60),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
        cv2.putText(frame_rgb,age_label,age_label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
        cv2.imwrite('recognize_all.jpg',frame_rgb)

    img = Image.fromarray(frame_rgb).resize((320,320))
    img_open = ImageTk.PhotoImage(image=img)
    lmain.imgtk = img_open
    lmain.configure(image=img_open)
    lmain.after(15,recognize_age)
#Write function stop_program  
def stop_program():
    quit()
#Write function display_result
def display_result():
    img1=cv2.imread('recognize_all.jpg')
    img = Image.fromarray(img1).resize((320,320))
    img_open = ImageTk.PhotoImage(image=img)
    lmain1.imgtk = img_open
    lmain1.configure(image=img_open)

while True:
    cap_btn =Button(root, text = 'Start',font=("Arial",14,"bold"),bd=3,bg="cyan",foreground="black",command=recognize_age)
    cap_btn.place(x=230,y=730)
    cap_btn1 =Button(root, text = 'Capture',font=("Arial",14,"bold"),bd=3,bg="cyan",foreground="black",command=display_result)
    cap_btn1.place(x=330,y=730)
    cap_btn2 =Button(root, text = 'Stop',font=("Arial",14,"bold"),bg="red",foreground="black",command=stop_program)
    cap_btn2.place(x=460,y=730)
    tit=Label(root,text='FINAL PROJECT AI',bd=3,bg='white',fg='blue',font=("Arial",20,"bold"))
    tit.place(x=240,y=0)
    tit=Label(root,text='RECOGNIZE GENDER,AGE AND EMOTION BASED ON FACE REAL-TIME',bd=3,bg='white',fg='blue',font=("Arial",16,"bold"))
    tit.place(x=10,y=30)
    root.mainloop()