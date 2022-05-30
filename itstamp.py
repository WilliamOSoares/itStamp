import cv2 as cv
import sys
import numpy as np
from pathlib import Path

# pegando o caminho da imagem
path = Path(sys.path[0])
caminhoImagem = str(path.parent.absolute()) + '\\#itstampa\\PrimeiroTeste\\camisa1.png'

# pegando a imagem e abrindo numa janela
imagem = cv.imread(caminhoImagem)
cv.namedWindow('Imagem Entrada',cv.WINDOW_GUI_EXPANDED)
cv.imshow('Imagem Entrada', imagem)

# Intervalo de branco para ser encontrado
low_white = np.array([200, 200, 200]) # B G R
high_white = np.array([255, 255, 255])
#hsv = cv.cvtColor(imagem, cv.COLOR_BGR2HSV)

# Manipulação da imagem para colocar a camisa em evidência
imagem1 = cv.cvtColor(imagem, cv.COLOR_BGR2GRAY)
mask1 = cv.inRange(imagem, low_white, high_white) 
#erode = cv.morphologyEx(mask1,cv.MORPH_ERODE,cv.getStructuringElement(cv.MORPH_CROSS,(3,3)))
res = cv.bitwise_and(imagem1, mask1)

cv.namedWindow('Imagem com foco na camisa',cv.WINDOW_GUI_EXPANDED)
cv.imshow('Imagem com foco na camisa', res)

cv.waitKey(0)

# Salva imagem
cv.imwrite(str(path.parent.absolute()) + "\\#itstampa\\PrimeiroTeste\\camisaSegm.png",res)