from __future__ import division
import cv2
import time
import numpy as np
import csv

#Arquivos da Biblioteca
protoFile = "hand/pose_deploy.prototxt"
weightsFile = "hand/pose_iter_102000.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

#Número de Pontos a serem detectados
nPoints = 21
POSE_PAIRS = [ [0,1,2,3],[2,3,4,5],[4,5,6,7],[6,7,8,9],[0,1,10,11],[10,11,12,13],[12,13,14,15],
[14,15,16,17],[0,1,18,19],[18,19,20,21],[20,21,22,23],[22,23,24,25],[0,1,26,27],
[26,27,28,29],[28,29,30,31],[30,31,32,33],[0,1,34,35],[34,35,36,37],[36,37,38,39],[38,39,40,41] ]

#Quantidade de Imagens a serem analisadas
qtdImagens = 50

for j in range(0, qtdImagens):
    cap = cv2.VideoCapture("http://192.168.137.20:4747/cam/1/frame.jpg")
    if( cap.isOpened() ) :
        ret,img = cap.read()
        #cv2.imshow("win",img)
        cv2.waitKey()
    frame = img
    frameCopy = np.copy(frame)
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    aspect_ratio = frameWidth/frameHeight

    threshold = 0.1

    t = time.time()
    # input image dimensions for the network
    inHeight = 368
    inWidth = int(((aspect_ratio*inHeight)*8)//8)
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)

    output = net.forward()
    print("time taken by network : {:.3f}".format(time.time() - t))

    # Empty list to store the detected keypoints
    points = []

    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]
        probMap = cv2.resize(probMap, (frameWidth, frameHeight))
        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        if prob > threshold :
            cv2.circle(frameCopy, (int(point[0]), int(point[1])), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
        # Add the point to the list if the probability is greater than the threshold
            points.append(int(point[0]))
            points.append(int(point[1]))
        else :
            points.append(0)
            points.append(0)
        
    # Draw Skeleton
    for pair in POSE_PAIRS:
        partAx = pair[0]
        partAy = pair[1]
        partBx = pair[2]
        partBy = pair[3]
        if points[partAx] and points[partAy] and points[partBx] and points[partBy]:
            cv2.line(frame, (points[partAx],points[partAy]), (points[partBx], points[partBy]), (0, 255, 255), 2)
            cv2.circle(frame, (points[partAx],points[partAy]), 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frame, (points[partBx], points[partBy]), 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

        #cv2.imshow('Output-Keypoints', frameCopy)
        #cv2.imshow('Output-Skeleton', frame)

    cv2.imwrite('Outputs/LetraU/Output-Keypoints_'+str(j)+'.jpg', frameCopy)
    cv2.imwrite('Outputs/LetraU/Output-Skeleton_'+str(j)+'.jpg', frame)

    print("Total time taken : {:.3f}".format(time.time() - t))

    print(points)

    #Criar arquivo .csv
    with open('Dados/keypointsLetraU.csv', mode='a') as employee_file:
        employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        employee_writer.writerow(points)

cv2.waitKey(0)
