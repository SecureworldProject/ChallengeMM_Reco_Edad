# Import Libraries
import cv2
import numpy as np
import math
import sys
import lock
import os
def get_face_box (net, frame, conf_threshold = 0.5):
  frame_copy = frame.copy()
  frame_height = frame_copy.shape[0]
  frame_width = frame_copy.shape[1]
  blob = cv2.dnn.blobFromImage(frame_copy, 1.0, (300, 300), [104, 117, 123], True, False)

  net.setInput(blob)
  detections = net.forward()
  boxes = []

  for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    if confidence > conf_threshold:
      x1 = int(detections[0, 0, i, 3] * frame_width)
      y1 = int(detections[0, 0, i, 4] * frame_height)
      x2 = int(detections[0, 0, i, 5] * frame_width)
      y2 = int(detections[0, 0, i, 6] * frame_height)
      boxes.append([x1, y1, x2, y2])
      cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), int(round(frame_height / 150)), 8)

  return frame_copy, boxes
def age_gender_detector (input_path):
  image = input_path
  resized_image = cv2.resize(image, (640, 480))

  frame = resized_image.copy()
  frame_face, boxes = get_face_box(FACE_NET, frame)
  edad=[]
  for box in boxes:
    face = frame[max(0, box[1] - box_padding):min(box[3] + box_padding, frame.shape[0] - 1), \
      max(0, box[0] - box_padding):min(box[2] + box_padding, frame.shape[1] - 1)]
    
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (104., 177., 123.), swapRB = False)
    """
    GENDER_NET.setInput(blob)
    gender_predictions = GENDER_NET.forward()
    gender = GENDER_LIST[gender_predictions[0].argmax()]
    print('bbbbb')
    print("Gender: {}, conf: {:.3f}".format(gender, gender_predictions[0].max()))
    print(gender)"""
    AGE_NET.setInput(blob)
    age_predictions = AGE_NET.forward()
    age = AGE_LIST[age_predictions[0].argmax()]
    print("Age: {}, conf: {:.3f}".format(age, age_predictions[0].max()))
    print(age)
    label = "{}".format( age)
    cv2.putText(frame_face, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    edad.append(age_predictions[0].argmax())

  return frame_face,edad

FACE_PROTO = "weights/opencv_face_detector.pbtxt"
FACE_MODEL = "weights/opencv_face_detector_uint8.pb"

AGE_PROTO = "weights/age_deploy.prototxt"
AGE_MODEL = "weights/age_net.caffemodel"

GENDER_PROTO = "weights/gender_deploy.prototxt"
GENDER_MODEL = "weights/gender_net.caffemodel"

# Load network
FACE_NET = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
AGE_NET = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
GENDER_NET = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_LIST = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)','(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']


GENDER_LIST = ["Hombre", "Mujer"]

box_padding = 20

props_dict = {}
DEBUG_MODE = True

def init(props):
    global props_dict
    print("Python: starting challenge init()")
    #cargamos el json que le pasemos y lo guardamos en la variable global
    props_dict = props
    return 0

def executeChallenge():

  print("Python: starting executeChallenge()")
  #comprobamos las variables de entorno y cogemos el de SECUREMIRROR_CAPTURES
  dataPath = os.environ['SECUREMIRROR_CAPTURES']
  print ("storage folder is :",dataPath)
  #abrimos lock
  lock.lockIN("reconocimiento_edad")
  


  cap = cv2.VideoCapture(0)

  while(True):
      ret, frame = cap.read()
      rgb = frame
      
  
      break

  cap.release()
  cv2.destroyAllWindows()
  del(cap)
  output,edad = age_gender_detector(rgb)
  #cerramos el lock
  lock.lockOUT("reconocimiento_edad")
  if max(edad)>2:
      cad=1
  else:
      cad=0
  cad="%d"%(cad)
  key = bytes(cad,'utf-8')
  key_size = len(key)
  result = (key, key_size)
  print("result:", result)

  cv2.waitKey(0)
  cv2.destroyAllWindows()

if __name__ == "__main__":
    executeChallenge()