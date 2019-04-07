# reads in a png image and binary thresholds it.

import numpy as np  # shortened (used often)
import cv2  # always need this
import sys

# image from command line arg
imgFile = cv2.imread(sys.argv[1], 0)
imgFile = cv2.resize(imgFile, (800, 1000))
imgFile = cv2.GaussianBlur(imgFile, (5, 5), 0)

ret, thImg1 = cv2.threshold(imgFile, 127, 255, cv2.THRESH_BINARY)

ret, thImg2 = cv2.threshold(imgFile, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

thImg3 = cv2.adaptiveThreshold(
    imgFile, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 2)

thImg4 = cv2.adaptiveThreshold(
    imgFile, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)

# stacking images side-by-side
hStack1 = np.hstack((imgFile, thImg1, thImg2, thImg3, thImg4))

fres = np.vstack((hStack1))  # stacking images side-by-side
cv2.imshow('Conveniently Stacked', fres)

cv2.waitKey(0)
cv2.destroyAllWindows()
