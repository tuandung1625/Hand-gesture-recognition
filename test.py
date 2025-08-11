<<<<<<< HEAD
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands = 1)
classifier = Classifier("Model/keras_model_updated.h5","Model/labels.txt")
offset = 20
imgSize = 200

counter = 0

with open("Model/labels.txt", "r") as f:
    labels = [line.strip().split(' ')[1] for line in f]

while  True:
  success, img = cap.read()
  
  if not success:
    break
  imgOutput = img.copy()
  hands, img = detector.findHands(img)
  if hands:
    hand = hands[0]
    x,y,w,h = hand['bbox']
    
    
    #Clamp values to image boundaries
    img_height, img_width = img.shape[:2]
    x1 = max(0, x - offset)
    y1 = max(0, y - offset)
    x2 = min(img_width, x + w + offset)
    y2 = min(img_height, y + h + offset)
    
    imgCrop = img[y1:y2, x1:x2]
    

    if imgCrop.size > 0:
      imgCropH, imgCropW = imgCrop.shape[:2]
      imgWhite = np.ones((imgSize, imgSize,3),np.uint8)*255
      
      aspectRatio = imgCropH / imgCropW
      
      if aspectRatio >1:
        k = imgSize / imgCropH
        wCal = math.ceil(k * imgCropW)
        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
        wGap = math.ceil((imgSize - wCal) / 2)
        imgWhite[:, wGap:wGap + wCal] = imgResize
        
        
      else:
        k = imgSize / imgCropW
        hCal = math.ceil(k * imgCropH)
        imgResize = cv2.resize(imgCrop, (imgSize, hCal))

        if hCal > imgSize:
            hCal = imgSize  # prevent overflow
            imgResize = cv2.resize(imgCrop, (imgSize, imgSize))  # fallback
            hGap = 0
        else:
            hGap = math.ceil((imgSize - hCal) / 2)

        imgWhite[hGap:hGap + hCal, :] = imgResize

      prediction, index = classifier.getPrediction(imgWhite, draw= False)
      print(prediction, index)
      
    cv2.putText(imgOutput, labels[index], (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255,0,255), 2)
    cv2.rectangle(imgOutput, (x-offset, y-offset), (x + imgCropW + offset, y + imgCropH + offset), (255, 0, 255), 4)
    cv2.imshow("ImageCrop", imgCrop)
    cv2.imshow("ImgWhite", imgWhite)
      
    
  cv2.imshow("Image", imgOutput)
  cv2.waitKey(1)
  
=======
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands = 1)
classifier = Classifier("Model/keras_model_updated.h5","Model/labels.txt")
offset = 20
imgSize = 200

counter = 0

with open("Model/labels.txt", "r") as f:
    labels = [line.strip().split(' ')[1] for line in f]

while  True:
  success, img = cap.read()
  
  if not success:
    break
  imgOutput = img.copy()
  hands, img = detector.findHands(img)
  if hands:
    hand = hands[0]
    x,y,w,h = hand['bbox']
    
    
    #Clamp values to image boundaries
    img_height, img_width = img.shape[:2]
    x1 = max(0, x - offset)
    y1 = max(0, y - offset)
    x2 = min(img_width, x + w + offset)
    y2 = min(img_height, y + h + offset)
    
    imgCrop = img[y1:y2, x1:x2]
    

    if imgCrop.size > 0:
      imgCropH, imgCropW = imgCrop.shape[:2]
      imgWhite = np.ones((imgSize, imgSize,3),np.uint8)*255
      
      aspectRatio = imgCropH / imgCropW
      
      if aspectRatio >1:
        k = imgSize / imgCropH
        wCal = math.ceil(k * imgCropW)
        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
        wGap = math.ceil((imgSize - wCal) / 2)
        imgWhite[:, wGap:wGap + wCal] = imgResize
        
        
      else:
        k = imgSize / imgCropW
        hCal = math.ceil(k * imgCropH)
        imgResize = cv2.resize(imgCrop, (imgSize, hCal))

        if hCal > imgSize:
            hCal = imgSize  # prevent overflow
            imgResize = cv2.resize(imgCrop, (imgSize, imgSize))  # fallback
            hGap = 0
        else:
            hGap = math.ceil((imgSize - hCal) / 2)

        imgWhite[hGap:hGap + hCal, :] = imgResize

      prediction, index = classifier.getPrediction(imgWhite, draw= False)
      print(prediction, index)
      
    cv2.putText(imgOutput, labels[index], (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255,0,255), 2)
    cv2.rectangle(imgOutput, (x-offset, y-offset), (x + imgCropW + offset, y + imgCropH + offset), (255, 0, 255), 4)
    cv2.imshow("ImageCrop", imgCrop)
    cv2.imshow("ImgWhite", imgWhite)
      
    
  cv2.imshow("Image", imgOutput)
  cv2.waitKey(1)
  
>>>>>>> 4c6101a (Save local files)
  