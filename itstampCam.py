import cv2 as cv
import sys
import numpy as np
from pathlib import Path
import time

cameraCapture = cv.VideoCapture(0)
inicio = time.time()
count = 0
# pegando o caminho da imagem
path = Path(sys.path[0])
caminhoImagem = str(path.parent.absolute()) + '\\itStamp\\imgEntradaCam\\'
caminhoResultado = str(path.parent.absolute()) + '\\itStamp\\imgSaidaCam\\'
while cameraCapture.isOpened() and cv.waitKey(1) == -1:
    sucess, frame = cameraCapture.read()    
    cv.imshow('Mostra webCam', frame)
    cv.imwrite(caminhoImagem + "frame%d.jpg" % count, frame)
    # pegando a imagem e abrindo numa janela
    imagem = cv.imread(caminhoImagem + "frame%d.jpg" % count)
    cv.namedWindow('Imagem Entrada',cv.WINDOW_AUTOSIZE)
    cv.imshow('Imagem Entrada', imagem)

    # Intervalo de branco para ser encontrado
    low_white = np.array([200, 200, 200]) # B G R
    high_white = np.array([255, 255, 255])
    #hsv = cv.cvtColor(imagem, cv.COLOR_BGR2HSV)

    # Manipulação da imagem para colocar a camisa em evidência
    imagem1 = cv.cvtColor(imagem, cv.COLOR_BGR2GRAY)
    mask1 = cv.inRange(imagem, low_white, high_white) 
    mask1 = cv.morphologyEx(mask1,cv.MORPH_ERODE,cv.getStructuringElement(cv.MORPH_CROSS,(3,3)))
    res = cv.bitwise_and(imagem1, mask1)#cv.cvtColor(mask1, cv.COLOR_GRAY2BGR))

    cv.namedWindow('Imagem com foco na camisa',cv.WINDOW_AUTOSIZE)
    cv.imshow('Imagem com foco na camisa', res)

    #Blob detection
    # Setup SimpleBlobDetector parameters.
    params = cv.SimpleBlobDetector_Params()

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 150

    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 200

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1
    
    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.87

    # Create a detector with the parameters
    ver = (cv.__version__).split('.')
    if int(ver[0]) < 3 :
        detector = cv.SimpleBlobDetector(params)
    else : 
        detector = cv.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(res)
    im_with_keypoints = cv.drawKeypoints(res, keypoints, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show blobs
    cv.namedWindow('Keypoints',cv.WINDOW_AUTOSIZE)
    cv.imshow("Keypoints", im_with_keypoints)
    cv.imwrite(caminhoResultado + "frame%d.jpg" % count, im_with_keypoints)


fim = time.time()
print("Tempo: " + str(int(round(fim - inicio, 0))) + " seg")
cv.waitKey(0)

cameraCapture.release()
