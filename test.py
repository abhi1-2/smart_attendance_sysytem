import numpy as np
import cv2
import dlib
import imutils
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from imutils import face_utils
# Capture video from file
cap = cv2.VideoCapture(0)
face_detector = dlib.get_frontal_face_detector()
predictor_model="/Users/avisheksarkar/Downloads/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_model)
facerec=dlib.face_recognition_model_v1('/Users/avisheksarkar/Downloads/dlib_face_recognition_resnet_model_v1.dat')
fa=FaceAligner(predictor, desiredFaceWidth=256)
id=6
sample_no=0
ret, frame = cap.read()
ret=True
while ret:
    
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = face_detector(gray, 1)
    
    if ret == True:
        
        
        for i, face_rect in enumerate(detected_faces):
            print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))
            print(face_rect)
            x=face_rect.left()
            y=face_rect.top()
            x2=face_rect.right()
            y2=face_rect.bottom()
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
            faceAligned = fa.align(frame, gray, face_rect)
            #shape=predictor(gray,face_rect)
            #face_descriptor = facerec.compute_face_descriptor(faceAligned, shape)
            #print(face_descriptor)
            #shape = face_utils.shape_to_np(shape)
            
            #for (x, y) in shape:
            #cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

            myface = frame[y:y2,x:x2]
    
       print( cv2.imwrite("/Users/avisheksarkar/Desktop/new1/dataSet/User"+str(id)+"/User"+str(id) +"."+ str(sample_no) + ".jpg",myface))
       sample_no=sample_no+1
        cv2.imshow('frame',faceAligned)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
        elif sample_no>20:
            break
    else:
      break
cap.release()
cv2.destroyAllWindows()


print("face detcted")

