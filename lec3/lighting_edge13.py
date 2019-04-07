# reads in a png image and detects edges. Also includes histogram equalization.

import numpy as np  # shortened (used often)
import cv2  # always need this
import sys

# image from command line arg
imgFile = cv2.imread(sys.argv[1], 0)
imgFile = cv2.resize(imgFile, (800, 1000))
# Create and move windows to set locations
cv2.namedWindow('Orig', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('gBlur', cv2.WINDOW_AUTOSIZE)

cv2.moveWindow("Orig", 150, 0)
cv2.moveWindow("gBlur", 0, 350)


cv2.imshow('Orig', imgFile)


gBlur = cv2.GaussianBlur(imgFile, (3, 3), 0)
#gBlur = imgFile

normImg = cv2.normalize(gBlur, None, 0, 255, cv2.NORM_MINMAX)
canny = cv2.Canny(normImg, 225, 250)  # tighter threshold
canny = cv2.Canny(normImg, 100, 200)
cv2.imshow('canny', canny)


equ = cv2.equalizeHist(gBlur)
canny2 = cv2.Canny(equ, 100, 200)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clImg = clahe.apply(gBlur)
canny3 = cv2.Canny(clImg, 100, 200)

hStack1 = np.hstack((imgFile, normImg, canny))  # stacking images side-by-side
hStack2 = np.hstack((imgFile, equ, canny2))  # stacking images side-by-side
hStack3 = np.hstack((imgFile, clImg, canny3))  # stacking images side-by-side

fres = np.vstack((hStack1, hStack2, hStack3))  # stacking images side-by-side
cv2.imshow('Conveniently Stacked', fres)

cv2.waitKey(0)
cv2.destroyAllWindows()
