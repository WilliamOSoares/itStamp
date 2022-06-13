import cv2 as cv
import sys
import numpy as np
from pathlib import Path

# pegando o caminho da imagem
path = Path(sys.path[0])
caminhoImagem = str(path.parent.absolute()) + '\\itStamp\\PrimeiroTeste\\camisaAlt.png'

# pegando a imagem e abrindo numa janela
imagem = cv.imread(caminhoImagem)
cv.namedWindow('Imagem Entrada',cv.WINDOW_AUTOSIZE)
cv.imshow('Imagem Entrada', imagem)

# Intervalo de branco para ser encontrado
low_green = np.array([20, 130, 200]) # B G R
high_green = np.array([120, 255, 255])
#hsv = cv.cvtColor(imagem, cv.COLOR_BGR2HSV)

# Manipulação da imagem para colocar a camisa em evidência
imagem1 = cv.cvtColor(imagem, cv.COLOR_BGR2GRAY)
mask1 = cv.inRange(imagem, low_green, high_green) 
mask1 = cv.morphologyEx(mask1,cv.MORPH_ERODE,cv.getStructuringElement(cv.MORPH_CROSS,(3,3)))
res = cv.bitwise_and(imagem1, mask1)#cv.cvtColor(mask1, cv.COLOR_GRAY2BGR))

cv.namedWindow('Imagem com foco na camisa',cv.WINDOW_AUTOSIZE)
cv.imshow('Imagem com foco na camisa', res)

cv.waitKey(0)

# Salva imagem
#cv.imwrite(str(path.parent.absolute()) + "\\itStamp\\PrimeiroTeste\\camisaSegm.png",res)