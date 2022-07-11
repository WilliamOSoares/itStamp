import cv2 as cv
import sys
import numpy as np
from pathlib import Path
import time

cameraCapture = cv.VideoCapture("http://192.168.15.2:8080/video")
inicio = time.time()
count = 0
# pegando o caminho da imagem
path = Path(sys.path[0])
caminhoImagem = str(path.parent.absolute()) + '\\itStamp\\imgEntradaCam\\'
caminhoResultado = str(path.parent.absolute()) + '\\itStamp\\imgSaidaCam\\'
caminhoEstampa = str(path.parent.absolute()) + '\\itStamp\\Estampas\\Pikachu.png'

# Intervalo de branco para ser encontrado
low_white = np.array([200, 200, 200]) # B G R
high_white = np.array([255, 255, 255])

while cameraCapture.isOpened() and cv.waitKey(1) == -1:
    sucess, frame = cameraCapture.read()    
    cv.imshow('Mostra webCam', frame)
    if(count % 30 == 0):
        cv.imwrite(caminhoImagem + "frame%d.jpg" % count, frame)
        # pegando a imagem e abrindo numa janela
        imagem = cv.imread(caminhoImagem + "frame%d.jpg" % count)
        #cv.namedWindow('Imagem Entrada',cv.WINDOW_AUTOSIZE)
        #cv.imshow('Imagem Entrada', imagem)

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
        lista =[]
        #Adicionando os pontos na lista
        for keyPoint in keypoints:
            x = keyPoint.pt[0]
            y = keyPoint.pt[1]
            s = keyPoint.size
            cord = (x,y)
            lista.append(cord)
        #Achando o tamanho da área:        
        rows,cols,_ = imagem.shape
        imgResize = np.zeros((rows,cols),np.uint8)
        pontoInit = []
        pontoLarg = []
        pontoAltu = []
        print(lista)
        for i in range(len(lista)):
            y,x = lista[i]
            if not pontoInit:
                pontoInit = x,y
                pontoLarg = x,y
                pontoAltu = x,y
            else:
                if(x > (pontoAltu[0]-10) and y< pontoAltu[1]):
                    pontoAltu = x,y
                if(x < pontoInit[0]):
                    pontoInit = x,y
                if(x == pontoInit[0] and y< pontoInit[1]):
                    pontoInit = x,y
                if(x < pontoLarg[0] and y>(pontoLarg[1]-10)):
                    pontoLarg = x,y

        imgResize[int(pontoInit[0]),int(pontoInit[1])] = 255
        imgResize[int(pontoLarg[0]),int(pontoLarg[1])] = 255
        imgResize[int(pontoAltu[0]),int(pontoAltu[1])] = 255
        cv.namedWindow("pontos de coordenada",cv.WINDOW_AUTOSIZE)
        cv.imshow("pontos de coordenada", imgResize)

        largInitX = pontoInit[1]
        largFimX = pontoLarg[1]
        altuInitY = pontoInit[0]
        altuFimY = pontoAltu[0]  
        larg = round(largFimX) - round(largInitX)
        altu = round(altuFimY) - round(altuInitY)

        # REMOVENDO OS PONTOS
        img = cv.cvtColor(im_with_keypoints, cv.COLOR_BGR2GRAY)
        contours, hierarchy = cv.findContours(img,cv.RETR_CCOMP,cv.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours)):
            cv.drawContours(imagem, contours, i, (255,255,255), cv.FILLED, 8)
        #cv.imshow("contorno", imagem)
        # COLANDO A ESTAMPA    
        estampa = cv.imread(caminhoEstampa)
        #Fazendo a sobreposição dos pixels da área da camisa e colocando a estampa
        resize = cv.resize(estampa, (int(larg),int(altu)))
        #cv.namedWindow('estampa',cv.WINDOW_AUTOSIZE)
        #cv.imshow("estampa", resize)

        for r in range(rows):
            for c in range(cols):
                if(r>=largInitX and r<=largFimX):            
                    if(c>= altuInitY and c<=altuFimY):
                        imagem[c,r] = resize[int(c-altuInitY), int(r-largInitX)]

        cv.namedWindow('Resultado',cv.WINDOW_AUTOSIZE)
        cv.imshow("Resultado", imagem)
        cv.imwrite(caminhoResultado + "frame.jpg", imagem)#cv.imwrite(caminhoResultado + "frame%d.jpg" % count, im_with_keypoints)


    count = count + 1
    

fim = time.time()
print("Tempo: " + str(int(round(fim - inicio, 0))) + " seg")
cv.waitKey(0)

cameraCapture.release()