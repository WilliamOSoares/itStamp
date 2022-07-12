from binascii import a2b_base64
from zipfile import LargeZipFile
import cv2 as cv
import sys
import numpy as np
from pathlib import Path
import time

# Marcação de tempo
inicio = time.time()

# pegando o caminho das imagens
path = Path(sys.path[0])
caminhoImagem = str(path.parent.absolute()) + '\\itStamp\\Teste\\camisa2.png'
caminhoEstampa = str(path.parent.absolute()) + '\\itStamp\\Estampas\\ecomp.png'

# pegando a imagem e abrindo numa janela
imagem = cv.imread(caminhoImagem)
cv.namedWindow('Imagem Entrada',cv.WINDOW_AUTOSIZE)
cv.imshow('Imagem Entrada', imagem)
cv.waitKey(0)

############################### Pré-processamento da imagem (segmentação da camisa) #####################
# Intervalo de branco para ser encontrado
low_white = np.array([180, 180, 180])
high_white = np.array([255, 255, 255])

# Manipulação da imagem para colocar a camisa em evidência
imagem1 = cv.cvtColor(imagem, cv.COLOR_BGR2GRAY)
mask1 = cv.inRange(imagem, low_white, high_white) 
mask1 = cv.morphologyEx(mask1,cv.MORPH_ERODE,cv.getStructuringElement(cv.MORPH_CROSS,(3,3)))
res = cv.bitwise_and(imagem1, mask1)
cv.namedWindow('Imagem com foco na camisa',cv.WINDOW_AUTOSIZE)
cv.imshow('Imagem com foco na camisa', res)
cv.waitKey(0)

################################# Blob detection ##########################################
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
cv.namedWindow("Blobs detectados",cv.WINDOW_AUTOSIZE)
cv.imshow("Blobs detectados", im_with_keypoints) #virou RGB
cv.waitKey(0)


############################# IDENTIFICANDO OS PONTOS #############################
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

#Identificando os pontos
for i in range(len(lista)):
    y,x = lista[i]
    if not pontoInit:
        pontoInit = x,y
        pontoLarg = x,y
        pontoAltu = x,y
    else:
        if(x > (pontoAltu[0]-20) and y< pontoAltu[1]):
            pontoAltu = x,y
        if(x == pontoInit[0] and y< pontoInit[1]):
            pontoInit = x,y
        if(x < pontoInit[0]):
            pontoInit = x,y        
        if(x < pontoLarg[0] and y>(pontoLarg[1]-20)):
            pontoLarg = x,y

imgResize[int(pontoInit[0]),int(pontoInit[1])] = 255
imgResize[int(pontoLarg[0]),int(pontoLarg[1])] = 255
imgResize[int(pontoAltu[0]),int(pontoAltu[1])] = 255
cv.imshow("Pontos de coordenada", imgResize)
cv.waitKey(0)

largInitX = pontoInit[1]
largFimX = pontoLarg[1]
altuInitY = pontoInit[0]
altuFimY = pontoAltu[0]  
larg = round(largFimX) - round(largInitX)
altu = round(altuFimY) - round(altuInitY)

############################# REMOVENDO OS PONTOS #####################################
img = cv.cvtColor(im_with_keypoints, cv.COLOR_BGR2GRAY)
contours, hierarchy = cv.findContours(img,cv.RETR_CCOMP,cv.CHAIN_APPROX_SIMPLE)
for i in range(len(contours)):
    cv.drawContours(imagem, contours, i, (200,200,200), cv.FILLED, 8)
cv.imshow("Removendo dos pontos", imagem)
cv.waitKey(0)

##################################### COLANDO A ESTAMPA #######################################  
estampa = cv.imread(caminhoEstampa)
#redimencionando as imagens
resize = cv.resize(estampa, (int(larg),int(altu)))
cv.namedWindow('Estampa redimencionada',cv.WINDOW_AUTOSIZE)
cv.imshow("Estampa redimencionada", resize)
cv.waitKey(0)

#Colocando a estampa na imagem
for r in range(rows):
    for c in range(cols):
        if(r>=largInitX and r<=largFimX):            
            if(c>= altuInitY and c<=altuFimY):
                imagem[c,r] = resize[int(c-altuInitY), int(r-largInitX)]

cv.namedWindow("Resultado Final",cv.WINDOW_AUTOSIZE)
cv.imshow("Resultado Final", imagem)

fim = time.time()
print("Tempo de execução: " + str(int(round(fim - inicio, 0))) + " seg")
cv.waitKey(0)

# Salva imagem
cv.imwrite(str(path.parent.absolute()) + "\\itStamp\\Teste\\resultado.png",imagem)