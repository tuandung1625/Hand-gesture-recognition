
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands = 1)

offset = 20
imgSize = 200

folder = "Data/C"
counter = 0

while  True:
  success, img = cap.read()
  
  if not success:
    break
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
        hGap = math.ceil((imgSize - hCal) / 2)
        imgWhite[hGap:hGap + hCal, :] = imgResize

    cv2.imshow("ImageCrop", imgCrop)
    cv2.imshow("ImgWhite", imgWhite)
      
    
  cv2.imshow("Image",img)
  key = cv2.waitKey(1)
  
  if key == ord("s"):
    counter += 1
    cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
    print(counter)