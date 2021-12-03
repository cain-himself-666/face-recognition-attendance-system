from imutils.video import VideoStream
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import face_recognition
import numpy as np
import argparse
import imutils
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
print(myList)
#fetch names from the image
for cl in myList:
    curlImg = cv2.imread(f'{path}/{cl}')
    images.append(curlImg)
    names.append(os.path.splitext(cl)[0])

print(names)
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
#starting the stream
video_capture = cv2.VideoCapture(0)
#looping over frames
while True:
    #checkpoint 1
    ret, frame = video_capture.read()

    if ret:
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
        rects = detector(gray, 0)
        frame = imutils.resize(frame, width=960, height=540)
        for rect in rects:
                
            x = rect.left()  
            y = rect.top()  
            x1 = rect.right()  
            y1 = rect.bottom()
            landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, rect).parts()])  
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

    (h, w) = frame.shape[:2]
    temp = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))
    net.setInput(temp)
    detections = net.forward()
        # read the face frames
    imgS = cv2.resize(frame, (0,0), None, 0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        # encode the face frames
    faceCurrentFrame = face_recognition.face_locations(imgS)
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
            face = frame[startY:endY, startX:endX]
            face = cv2.resize(face, (32, 32))
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
                    # print(faceDistance)
                    matchIndex = np.argmin(faceDistance)
                    if matches[matchIndex]:
                        name = names[matchIndex].upper()
                        text = "{}".format(name)
                        cv2.putText(frame, text, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.rectangle(frame, (startX, startY), (endX, endY),
                            (0,255,0), 2)
                    else:
                        text = "{}".format("User Not Registered !!")
                        cv2.putText(frame, text, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.rectangle(frame, (startX, startY), (endX, endY),
                            (0,0,255), 2)
                    
            else:
                warning = "Warning !! Fake Photo"
                cv2.putText(frame, warning, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                    (0,0,255), 2)

    # def show_frame():
    #     flip_frame = cv2.flip(frame,1)
    #     cv2image = cv2.cvtColor(flip_frame, cv2.COLOR_BGR2RGBA)
    #     imgtk = ImageTk.PhotoImage(image=Image.fromarray(cv2image))
    #     lmain.imgtk = imgtk
    #     lmain.configure(image=imgtk)
    #     lmain.after(10, show_frame) 

    # show_frame()
    # root.mainloop()
    cv2.imshow("Attendance System", frame)
    cv2.waitKey(1)
    

#vs.stop()





