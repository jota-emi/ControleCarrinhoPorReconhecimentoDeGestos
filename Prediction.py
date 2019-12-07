from __future__ import division
import cv2
import time
import numpy as np
import csv
import pandas as pd
from scipy.spatial import distance
import requests

protoFile = "hand/pose_deploy.prototxt"
weightsFile = "hand/pose_iter_102000.caffemodel"
nPoints = 21
POSE_PAIRS = [ [0,1,2,3],[2,3,4,5],[4,5,6,7],[6,7,8,9],[0,1,10,11],[10,11,12,13],[12,13,14,15],
[14,15,16,17],[0,1,18,19],[18,19,20,21],[20,21,22,23],[22,23,24,25],[0,1,26,27],
[26,27,28,29],[28,29,30,31],[30,31,32,33],[0,1,34,35],[34,35,36,37],[36,37,38,39],[38,39,40,41] ]
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

#Quantidade de vezes que o programa deve rodar
qtdImagens = 15

#IP do ESP8266 - Para controle do carrinho
ipArduino = 'http://192.168.137.203/'

#IP do DroidCam (De onde o programa deve capturar as imagens)
ipCam = "http://192.168.137.113:4747/cam/1/frame.jpg"

for j in range(1, qtdImagens+1):
    cap = cv2.VideoCapture(ipCam)
    if( cap.isOpened() ) :
        ret,img = cap.read()
        print("IMAGEM CAPTURADA")
        
        #cv2.imshow("win",img)
        cv2.waitKey()
    frame = img
    frameCopy = np.copy(frame)
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    aspect_ratio = frameWidth/frameHeight

    threshold = 0.1

    t = time.time()

    #Dimensões da imagem de entrada
    inHeight = 368
    inWidth = int(((aspect_ratio*inHeight)*8)//8)
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)

    output = net.forward()
    #print("time taken by network : {:.3f}".format(time.time() - t))

    #Lista vazia para guardar os keypoints detectados
    points = []

    for i in range(nPoints):
        #confidence map of corresponding body's part.
        probMap = output[0, i, :, :]
        probMap = cv2.resize(probMap, (frameWidth, frameHeight))

        #Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        if prob > threshold :
            cv2.circle(frameCopy, (int(point[0]), int(point[1])), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
            # Adicionando os pontos da imagem (em pixels) a lista
            points.append(int(point[0]))
            points.append(int(point[1]))
        else :
            points.append(0)
            points.append(0)
    
    # Desenhar Esqueleto
    for pair in POSE_PAIRS:
        partAx = pair[0]
        partAy = pair[1]
        partBx = pair[2]
        partBy = pair[3]
        if points[partAx] and points[partAy] and points[partBx] and points[partBy]:
            cv2.line(frame, (points[partAx],points[partAy]), (points[partBx], points[partBy]), (0, 255, 255), 2)
            cv2.circle(frame, (points[partAx],points[partAy]), 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frame, (points[partBx], points[partBy]), 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

    
    
    #Imprimir Tempo total de execução e posição dos pontos encontrados
    print("Total time taken : {:.3f}".format(time.time() - t))
    print(points)

    #Obter database que contém a rede neural
    neuronios = pd.read_csv("keypointsSOMTreinada.csv")
    x = neuronios.iloc[:, :-1].values
    y = neuronios.iloc[:,-1]

    distancias = []
    neuroVenc = 0

    #Encontrar neurônio vencedor 
    for i in range (len(x)):
        distancias.append(distance.euclidean(points,x[i]))
    neuroVenc = distancias.index(min(distancias))
    print("Neurônio Vencedor: " + str(neuroVenc))

    #Imprimir resposta da rede para o novo dado e mandar requisição para o ESP, se for o caso
    if(y[neuroVenc] == 0):
        #r = requests.get(ip + 'S')
        print("LETRA 'A'\n")
    elif(y[neuroVenc] == 1):
        #r = requests.get(ip + 'F')
        print("LETRA 'E'\n")
    elif(y[neuroVenc] == 2):
        #r = requests.get(ip + 'R')
        print("LETRA 'I'\n")
    elif(y[neuroVenc] == 3):
        #r = requests.get(ip + 'L')
        print("LETRA 'O'\n")
    elif(y[neuroVenc] == 4):
        #r = requests.get(ip + 'B')
        print("LETRA 'U'\n")

    #Mostrar na tela as imagens de saída
    cv2.imshow('Output-Keypoints', frameCopy)
    cv2.imshow('Output-Skeleton', frame)

    #cv2.destroyAllWindows()

cv2.waitKey(0)