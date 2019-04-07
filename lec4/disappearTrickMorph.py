import numpy as np
import cv2
import sys


def drawMinEnclose(resized, circles):
    (x, y), radius = cv2.minEnclosingCircle(circles)
    center = (int(x), int(y))
    radius = int(radius)
    cv2.circle(resized, center, radius, (0, 255, 0), 2)


# image from command line arg
imgFile = cv2.imread(sys.argv[1], 1)
cv2.imshow('Original', imgFile)

#resized = cv2.resize(imgFile, (500, 500))
#resized = cv2.resize(imgFile, (800, 1000))
resized = cv2.resize(imgFile, (1600, 2000))
#resized = imgFile.copy()

gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

gray_blur = cv2.GaussianBlur(gray, (11, 11), 0)
#thresh = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 1)
ret, thresh = cv2.threshold(
    gray_blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

kernel = np.ones((3, 3), np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
                           kernel, iterations=1)

cont_img = closing.copy()
contours, hierarchy = cv2.findContours(
    cont_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

black = np.zeros(gray.shape)

for circles in contours:
    area = cv2.contourArea(circles)
    if area < 100 or area > 350:
        continue

    if len(circles) < 5:
        continue

    ellipse = cv2.fitEllipse(circles)
    #cv2.ellipse(resized, ellipse, (0,255,0), 2)
    drawMinEnclose(resized, circles)
    cv2.ellipse(black, ellipse, (255, 255, 255), -1, 2)

byteMask = np.asarray(black, dtype=np.uint8)
holes = cv2.bitwise_and(gray, byteMask)

resized = cv2.resize(resized, (800, 1000))
cv2.imshow("OTSU Thresholding", thresh)
cv2.imshow("Morphological Closing", closing)
cv2.imshow('Contours', resized)

# cv2.imshow('byteMask',byteMask)
# cv2.imshow('holes',holes)

cv2.waitKey(0)

cv2.destroyAllWindows()
