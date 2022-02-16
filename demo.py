from keras.preprocessing.image import img_to_array
from keras.models import load_model
import psycopg2
from playsound import playsound
import tkinter
from datetime import datetime
from PIL import Image, ImageTk
import face_recognition
import numpy as np
import argparse
import pickle
import time
import cv2
import os
import dlib
from scipy.spatial import distance as dist



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
x = 0
label =''

# construct argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
	help="path to trained model")
ap.add_argument("-l", "--le", type=str, required=True,
	help="path to label encoder")
ap.add_argument("-d", "--detector", type=str, required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
args = vars(ap.parse_args())

# loading face detector from the place where we stored it
print("loading face detector")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
#Loading the caffe model 
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
#reading data from the model.
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# loading the liveness detecting module that was trained in the training python script
print("loading the liveness detector")
model = load_model(args["model"])
le = pickle.loads(open(args["le"], "rb").read())


#determining the facial points that are plotted by dlib
FULL_POINTS = list(range(0, 68))  
FACE_POINTS = list(range(17, 68))  
   
JAWLINE_POINTS = list(range(0, 17))  
RIGHT_EYEBROW_POINTS = list(range(17, 22))  
LEFT_EYEBROW_POINTS = list(range(22, 27))  
NOSE_POINTS = list(range(27, 36))  
RIGHT_EYE_POINTS = list(range(36, 42))  
LEFT_EYE_POINTS = list(range(42, 48))  
MOUTH_OUTLINE_POINTS = list(range(48, 61))  
MOUTH_INNER_POINTS = list(range(61, 68))  
   
EYE_AR_THRESH = 0.30 
EYE_AR_CONSEC_FRAMES = 2  

#initializing the parameters
COUNTER_LEFT = 0  
TOTAL_LEFT = 0  
   
COUNTER_RIGHT = 0  
TOTAL_RIGHT = 0 
#defining a function for calculating ear and then comparing with the confidence parametrs

def eye_aspect_ratio(eye):
    
    A = dist.euclidean(eye[1], eye[5])  
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])  
    ear = (A + B) / (2.0 * C)  
    return ear 

#loading the predictor for predicting
detector = dlib.get_frontal_face_detector()  
#accessing the shape predictor
predictor = dlib.shape_predictor(args["shape_predictor"])
#fetch images from Image Library
path = "ImageLibrary"
images = []
names = []
myList = os.listdir(path)
#fetch names from the image
for cl in myList:
    curlImg = cv2.imread(f'{path}/{cl}')
    images.append(curlImg)
    labels = os.path.splitext(cl)[0]
    if labels.startswith('.'):
        continue
    names.append(labels.split("_")[0])

#find encodings of the images
def findEncodings(images):
    encodingList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodingList.append(encode)
    return encodingList

encodeListKnown = findEncodings(images)
print('Encoding Complete')
# Load an color image
root = tkinter.Tk()
root.title('HCS Attendance')
root.geometry("900x800")
heading = tkinter.Label(root,text="High Court of Sikkim Attendance System")
heading.config(font = ("Helvetica", 28))
heading.pack()

text1 = tkinter.Label(root, text="Rules to follow: ")
text1.config(font = ("Helvetica", 20))
text1.pack()

text2 = tkinter.Label(root, text="1) Please remove your mask for attendance")
text2.config(font = ("Helvetica", 20))
text2.pack()

text3 = tkinter.Label(root, text="2) One face at a time for attendace marking")
text3.config(font = ("Helvetica", 20))
text3.pack()

imageFrame = tkinter.Frame(root, width=700, height=600)
imageFrame.grid(row=0, column=0, padx=10, pady=2)
imageFrame.pack()

frame = tkinter.Frame(root, bg="black")
frame.place(relheight=0.6, relwidth=0.6, relx=0.2, rely=0.2)

lmain = tkinter.Label(frame)
lmain.grid(row=0, column=0)

response = tkinter.Label(root)
response.config(font = ("Helvetica", 20))
response.pack()

#Mark Attendance
def markInTime(name, date, intime):
    id = ''
    try:
        conn = psycopg2.connect(
            host="10.182.144.233",
            database="oss",
            user="postgres",
            password="Hcs@2021!",
            port="5432"
        )
        cur = conn.cursor()
        fullname_query = """SELECT employee_name from api_userprofiles WHERE employee_username='%s'""" % name
        cur.execute(fullname_query)
        fullname = cur.fetchone()
        query = """SELECT EXISTS(SELECT * FROM api_attendance WHERE employee_username_id=%s AND date_entry=%s)"""
        params = (name, date)
        cur.execute(query,params)
        result = cur.fetchone()
        if not result[0]:
            get_id_query = """SELECT * FROM api_attendance ORDER BY id DESC"""
            get_id_params = (name,date)
            cur.execute(get_id_query,get_id_params)
            init_id = cur.fetchone()
            # print(init_id[0])
            if not init_id:
                id = '1'
            else:
                id = init_id[0]+1

            intime_query = """INSERT INTO api_attendance(id, employee_username_id, date_entry, in_time) VALUES(%s,%s,%s,%s)"""
            intime_params = (id,name,date,intime)
            cur.execute(intime_query,intime_params)
            response['text'] = "Welcome {}. Your In Time has been marked on {} at {} AM".format(fullname[0], date,intime)
            response['foreground'] = "green"
        else:
            response['text'] = "Your In Time has been already been marked"
            response['foreground'] = "green"
            pass
        conn.commit()
        
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)

def markOutTime(name, date, outtime):
    try:
        conn = psycopg2.connect(
            host="10.182.144.233",
            database="oss",
            user="postgres",
            password="Hcs@2021!",
            port="5432"
        )
        cur = conn.cursor()
        fullname_query = """SELECT employee_name from api_userprofiles WHERE employee_username='%s'""" % name
        cur.execute(fullname_query)
        fullname = cur.fetchone()
        query = """SELECT EXISTS(SELECT * FROM api_attendance WHERE employee_username_id=%s AND date_entry=%s)"""
        params = (name, date)
        cur.execute(query,params)
        result = cur.fetchone()
        if result[0]:
            outtime_query = """UPDATE api_attendance SET out_time=%s WHERE employee_username_id=%s AND date_entry=%s"""
            outtime_params = (outtime,name,date)
            cur.execute(outtime_query,outtime_params)
            response['text'] = "Thank You {}. Your Out Time has been marked on {} at {} PM".format(fullname[0], date,outtime)
            response['foreground'] = "green"
        else:
            response['text'] = "Your Out Time has been already been marked"
            response['foreground'] = "green"
            pass
        conn.commit()
        
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
                
#starting the stream
video_capture = cv2.VideoCapture(0)
video_capture.set(3, 640)
video_capture.set(4, 480)
scale = 32
#looping over frames
while True:
    #checkpoint 1
    ret, frame = video_capture.read()
    flip_frame = cv2.flip(frame, 1)
    (h, w) = flip_frame.shape[:2]
    #prepare the crop
    centerX,centerY=int(h/2),int(w/2)
    radiusX,radiusY= int(scale*h/100),int(scale*w/100)
    minX,maxX=centerX-radiusX,centerX+radiusX
    minY,maxY=centerY-radiusY,centerY+radiusY
    cropped = flip_frame[minX:maxX, minY:maxY]
    resized_cropped = cv2.resize(cropped, (w, h))
    if ret:    
        gray = cv2.cvtColor(resized_cropped, cv2.COLOR_BGR2GRAY)  
        rects = detector(gray, 0)
        for rect in rects:      
            x = rect.left()  
            y = rect.top()  
            x1 = rect.right()  
            y1 = rect.bottom()
            landmarks = np.matrix([[p.x, p.y] for p in predictor(resized_cropped, rect).parts()])  
            left_eye = landmarks[LEFT_EYE_POINTS]  
            right_eye = landmarks[RIGHT_EYE_POINTS]  
            left_eye_hull = cv2.convexHull(left_eye)  
            right_eye_hull = cv2.convexHull(right_eye)  
            ear_left = eye_aspect_ratio(left_eye)  
            ear_right = eye_aspect_ratio(right_eye) 
            #calculating blink wheneer the ear value drops down below the threshold
            if ear_left < EYE_AR_THRESH:
                    
                COUNTER_LEFT += 1
                
            else:
                    
                    
                if COUNTER_LEFT >= EYE_AR_CONSEC_FRAMES:
                        
                    TOTAL_LEFT += 1  
                    COUNTER_LEFT = 0

            if ear_right < EYE_AR_THRESH:  
                    
                    
                COUNTER_RIGHT += 1  

            else:
                    
                if COUNTER_RIGHT >= EYE_AR_CONSEC_FRAMES: 
                        
                    TOTAL_RIGHT += 1   
                    COUNTER_RIGHT = 0


            x = TOTAL_LEFT + TOTAL_RIGHT

    temp = cv2.dnn.blobFromImage(cv2.resize(resized_cropped, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))
    net.setInput(temp)
    detections = net.forward()
    # read the face frames
    imgS = cv2.resize(resized_cropped, (0,0), None, 0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    # encode the face frames
    faceCurrentFrame = face_recognition.face_locations(imgS)
    if not faceCurrentFrame:
        response["text"]=""
    encodeCurrentFrame = face_recognition.face_encodings(imgS, faceCurrentFrame)
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]        
        #staisfying the union need of veryfying through ROI and blink detection.  
        if confidence > args["confidence"] and x>10:    
            #detect a bounding box
            #take dimensions
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            #get the dimensions
            (startX, startY, endX, endY) = box.astype("int")
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            # extract the face ROI and then preproces it in the exact
            # same manner as our training data
            face = resized_cropped[startY:endY, startX:endX]
            face = cv2.resize(face,(32,32))
            face = face.astype("float") / 255.0
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)

        #pass the model to determine the liveness
            preds = model.predict(face)[0]
            j = np.argmax(preds)
            label = le.classes_[j]
            if label == "real":
                for faceLoc,encodeFace in zip(faceCurrentFrame, encodeCurrentFrame):
                    matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                    faceDistance = face_recognition.face_distance(encodeListKnown, encodeFace)
                    matchIndex = np.argmin(faceDistance)

                    if matches[matchIndex]:
                        if faceDistance[matchIndex] < 0.48:
                            timedate = datetime.now()
                            currentDate = timedate.strftime('%Y-%m-%d')
                            currentTime12Hr = timedate.strftime('%I:%M:%S')
                            currentTime24Hr = timedate.strftime('%H:%M:%S')
                            name = names[matchIndex].lower()
                            cv2.putText(resized_cropped, '', (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            cv2.rectangle(resized_cropped, (startX, startY), (endX, endY),
                                (0,255,0), 2)
                            if currentTime24Hr > '07:00' and currentTime24Hr < '11:30':
                                markInTime(name, currentDate ,currentTime12Hr)
                            if currentTime24Hr > '15:30' and currentTime24Hr < '21:00':
                                markOutTime(name, currentDate ,currentTime12Hr)
                        else:
                            cv2.putText(resized_cropped, 'Unknown', (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            cv2.rectangle(resized_cropped, (startX, startY), (endX, endY),
                                (0,0,255), 2)
                    else:
                        cv2.putText(resized_cropped, 'Unknown', (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.rectangle(resized_cropped, (startX, startY), (endX, endY),
                            (0,0,255), 2)
                    

            else:
                cv2.putText(resized_cropped, 'Unknown', (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.rectangle(resized_cropped, (startX, startY), (endX, endY),
                    (0,0,255), 2)
                response['text'] = "Warning !!! Please do not display photo from any kind of device or remove your mask"
                response['foreground'] = "red"

    cv2image = cv2.cvtColor(resized_cropped, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain['image'] = imgtk

    root.update()
    

#vs.stop()





