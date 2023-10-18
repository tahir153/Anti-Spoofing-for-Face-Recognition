# import cvzone
# from cvzone.FaceDetectionModule import FaceDetector
# import cv2
# import time
#
# ####################
# classID = 0 # 0 for fake and 1 for real
# outputFolderPath = 'Dataset/DataCollect'
# confidence = 0.8
# save = True
# debug = False
# blurThreshold = 35 # Larger is more focus
# offsets_PercentageW = 10
# offsets_PercentageH = 20
# camWidth, camHeight = 640, 480
# floatingpoint = 6
# ####################
#
# cap = cv2.VideoCapture(0)
# cap.set(3,camWidth)
# cap.set(4,camHeight)
# detector = FaceDetector()
# while True:
#     success, img = cap.read()
#     imgOut = img.copy()
#     img, bboxs = detector.findFaces(img, draw=False)
#
#     listBlur = [] # True False value indicating if the faces are blur or not
#     listInfo = [] # The normalized values and the class name for the label txt file
#
#     if bboxs:
#         for bbox in bboxs:
#             x, y, w, h = bbox['bbox']
#             score = bbox['score'][0]
#             # print(x,y,w,h)
#             #------------- Check the Score ---------
#             if score>confidence:
#
#                 #--------Adding offsets to the face detected bounding boxes ----------
#                 offsetW = (offsets_PercentageW / 100)*w
#                 x = int(x - offsetW)
#                 w = int(w + offsetW * 2)
#                 offsetH = (offsets_PercentageH / 100) * h
#                 y = int(y - offsetH * 3)
#                 h = int(h + offsetH * 3.5)
#                 # --------- To avoid values below 0 -------
#                 if x < 0: x = 0
#                 if y < 0: y = 0
#                 if w < 0: w = 0
#                 if h < 0: h = 0
#                 #--------- Find blurriness-------
#                 imgFace = img[y:y+h, x:x+w]
#                 cv2.imshow("Face",imgFace)
#                 BlurValue = int(cv2.Laplacian(imgFace,cv2.CV_64F).var())
#                 if BlurValue>blurThreshold:
#                     listBlur.append(True)
#                 else:
#                     listBlur.append(False)
#
#
#
#
#                 # --------- Normalize Values -------
#                 ih, iw, _ = img.shape
#                 xc, yc = x+w/2 , y+h/2
#                 xcn, ycn = round(xc/iw,floatingpoint), round(yc/ih,floatingpoint)
#                 wn, hn = round(w/iw,floatingpoint), round(h/ih,floatingpoint)
#                 # print(xcn,ycn,wn, hn)
#                 # --------- To avoid values above 1 -------
#                 if xcn < 1: xcn = 1
#                 if ycn < 1: ycn = 1
#                 if wn < 1: wn = 1
#                 if hn < 1: hn = 1
#
#                 listInfo.append(f'{classID} {xcn} {ycn} {wn} {hn} \n')
#                 # --------- Drawing -------
#                 cv2.rectangle(imgOut, (x, y, w, h), (255,0,0), 3)
#                 cvzone.putTextRect(imgOut,f'Score:{int(score*100)} % Blur: {BlurValue}',(x,y-20), scale=1,thickness=2)
#                 if debug:
#                     cv2.rectangle(img, (x, y, w, h), (255, 0, 0), 3)
#                     cvzone.putTextRect(img, f'Score:{int(score * 100)} % Blur: {BlurValue}', (x, y - 20), scale=1,thickness=2)
#
#         # --------- to save -------
#         if save:
#             if all(listBlur) and listBlur!=[]:
#                 # --------- save image -------
#                 timeNow = time.time()
#                 timeNow = str(timeNow).split('.')
#                 timeNow = timeNow[0] + timeNow[1]
#                 # print(time.time())
#                 print(timeNow)
#                 cv2.imwrite(f'{outputFolderPath}/{timeNow}.jpg',img)
#                 # --------- save label text file -------
#                 for info in listInfo:
#                     f = open(f'{outputFolderPath}/{timeNow}.txt', 'a')
#                     f.write(info)
#                     f.close()
#
#
#     cv2.imshow("Image", imgOut)
#     cv2.waitKey(1)



####################################################################################################


import cvzone
from cvzone.FaceDetectionModule import FaceDetector
import cv2
import time

####################
classID = 0  # 0 for fake and 1 for real
outputFolderPath = 'Dataset/DataCollect'
confidence = 0.8
save = True
debug = False
blurThreshold = 35  # Larger is more focus
offsets_PercentageW = 10
offsets_PercentageH = 20
camWidth, camHeight = 640, 480
floatingpoint = 6
####################

cap = cv2.VideoCapture(0)
cap.set(3, camWidth)
cap.set(4, camHeight)
detector = FaceDetector()
while True:
    success, img = cap.read()
    imgOut = img.copy()
    img, bboxs = detector.findFaces(img, draw=False)

    listBlur = []  # True False value indicating if the faces are blur or not
    listInfo = []  # The normalized values and the class name for the label txt file

    if bboxs:
        for bbox in bboxs:
            x, y, w, h = bbox['bbox']
            score = bbox['score'][0]
            # print(x,y,w,h)
            # ------------- Check the Score ---------
            if score > confidence:

                # --------Adding offsets to the face detected bounding boxes ----------
                offsetW = (offsets_PercentageW / 100) * w
                x = int(x - offsetW)
                w = int(w + offsetW * 2)
                offsetH = (offsets_PercentageH / 100) * h
                y = int(y - offsetH * 3)
                h = int(h + offsetH * 3.5)
                # --------- To avoid values below 0 -------
                if x < 0: x = 0
                if y < 0: y = 0
                if w < 0: w = 0
                if h < 0: h = 0
                # --------- Find blurriness-------
                imgFace = img[y:y + h, x:x + w]
                cv2.imshow("Face", imgFace)
                BlurValue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())
                if BlurValue > blurThreshold:
                    listBlur.append(True)
                else:
                    listBlur.append(False)

                # --------- Normalize Values -------
                ih, iw, _ = img.shape
                xc, yc = x + w / 2, y + h / 2
                xcn, ycn = round(xc / iw, floatingpoint), round(yc / ih, floatingpoint)
                wn, hn = round(w / iw, floatingpoint), round(h / ih, floatingpoint)
                # --------- To avoid values above 1 -------
                if xcn > 1: xcn = 1
                if ycn > 1: ycn = 1
                if wn > 1: wn = 1
                if hn > 1: hn = 1

                listInfo.append(f'{classID} {xcn} {ycn} {wn} {hn} \n')
                # --------- Drawing -------
                cv2.rectangle(imgOut, (x, y, w, h), (255, 0, 0), 3)
                cvzone.putTextRect(imgOut, f'Score:{int(score * 100)} % Blur: {BlurValue}', (x, y - 20), scale=1, thickness=2)
                if debug:
                    cv2.rectangle(img, (x, y, w, h), (255, 0, 0), 3)
                    cvzone.putTextRect(img, f'Score:{int(score * 100)} % Blur: {BlurValue}', (x, y - 20), scale=1, thickness=2)

        # --------- to save -------
        if save:
            if all(listBlur) and listBlur != []:
                # --------- save image -------
                timeNow = time.time()
                timeNow = str(timeNow).split('.')
                timeNow = timeNow[0] + timeNow[1]
                # print(time.time())
                print(timeNow)
                cv2.imwrite(f'{outputFolderPath}/{timeNow}.jpg', img)
                # --------- save label text file -------
                for info in listInfo:
                    f = open(f'{outputFolderPath}/{timeNow}.txt', 'a')
                    f.write(info)
                    f.close()

    cv2.imshow("Image", imgOut)
    cv2.waitKey(1)
















