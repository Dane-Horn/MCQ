#reads in a png image and detects edges. Also includes histogram equalization.

import numpy as np #shortened (used often)
import cv2 # always need this
import sys

def auto_canny(image, sigma=0.9):
	# median of grey pixel intensities
	# tuning the sigma constant is optional
	v = np.median(image)
 
	# apply Canny edge detection using computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper) 

	return edged

# image from command line arg
imgFile = cv2.imread(sys.argv[1],0)


gBlur = cv2.GaussianBlur(imgFile, (5, 5), 0)
#gBlur = imgFile

normImg = cv2.normalize(gBlur, None, 0, 255, cv2.NORM_MINMAX)
canny = cv2.Canny(normImg,100,200)

equ = cv2.equalizeHist(gBlur)
canny2 = cv2.Canny(equ,100,200)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clImg = clahe.apply(gBlur)
canny3 = cv2.Canny(clImg,100,200)

autoEdgeImg = auto_canny(normImg)

hStack1 = np.hstack((imgFile,canny,canny2,canny3,autoEdgeImg)) #stacking images side-by-side

fres = np.vstack((hStack1)) #stacking images side-by-side
cv2.imshow('Conveniently Stacked',fres)

cv2.waitKey(0)
cv2.destroyAllWindows()
