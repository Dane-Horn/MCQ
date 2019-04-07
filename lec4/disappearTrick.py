import cv2
import numpy as np
import sys

# image from command line arg
imgFile = cv2.imread(sys.argv[1],1)
imgFile = cv2.resize(imgFile,(500,500))

#to grey
grey = cv2.cvtColor(imgFile, cv2.COLOR_BGR2GRAY)
cv2.imshow('greyOrig',grey)

#*****************************************************************************************************
#The hardcoded aka not so good way:

# location and size of the circle
xc, yc, r = 216, 245, 12
xc2, yc2, r2 = 247, 274, 12
# height and width of image
h,w = grey.shape # yes OpenCV reads Mat images backwards: rows first (y) followed by columns (x)
# coordinates per every pixel of the image stored into arrays x and y
x, y = np.meshgrid(np.arange(w), np.arange(h))
# squared distance from the center of the circle

d = (x - xc)**2 + (y - yc)**2
d2 = (x - xc2)**2 + (y - yc2)**2

# mask is a boolean array, and True inside of the circle
#seems to automatically append to the boolean array like a loop
mask = d < r**2
mask2 = d2 < r2**2

# we need to convert the mask to 8bit single channel image
badByteMask = np.asarray(mask*255, dtype=np.uint8)
#cv2.imshow('badByteMask',badByteMask)
badByteMask += np.asarray(mask2*255, dtype=np.uint8)
#cv2.imshow('badByteMask',badByteMask)
badByteMask_inv = cv2.bitwise_not(badByteMask)
#cv2.imshow('NOT op: badByteMask',badByteMask_inv)
addImage = cv2.add(grey,badByteMask)
#cv2.imshow('Add (superimpose)',addImage)
badByteMask_and = cv2.bitwise_and(grey, badByteMask)
addWeighted = cv2.addWeighted(grey,0.2,badByteMask,0.8,0)
cv2.imshow('addWeighted',addWeighted)
#cv2.imshow('AND op: badByteMask',badByteMask_and)


inpainted = cv2.inpaint(grey, badByteMask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

cv2.imshow('Inpainted',inpainted)


#hStack1 = np.hstack((grey,badByteMask,inpainted)) #stacking images side-by-side
#fres = np.vstack((hStack1)) #stacking images side-by-side
#cv2.imshow('Bad Method',fres)

#*****************************************************************************************************
#The better way:

grey = cv2.GaussianBlur(grey, (3,3), 0)

#ret,grey = cv2.threshold(grey,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#grey = cv2.adaptiveThreshold(grey,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,3,2)
#cv2.imshow('Threshold',grey)

circles = cv2.HoughCircles(grey, cv2.HOUGH_GRADIENT, dp=11, minDist=1, minRadius=5, maxRadius=18)
blue = (0,0,255)

color_img = grey.copy() # perfect copy in new memory space
#color_img = cv2.copyMakeBorder(imgFile,0,0,0,0,cv2.BORDER_REPLICATE) # copy with padding

cntCircles = 0


for x, y, radius in circles[0]:
    cv2.circle(color_img, (x,y), radius, blue, 2)
    
    black = np.zeros(grey.shape)
for x, y, radius in circles[0]:
    cv2.circle(black, (x,y), int(radius+5), 255, -1)  # Extra 4 to radius... -1 draws coloured-in circles
    cntCircles+=1  

print("There are ",cntCircles, "circles")    

goodByteMask = np.asarray(black, dtype=np.uint8)

inpainted2 = cv2.inpaint(color_img, goodByteMask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

cv2.imshow('goodByteMask',goodByteMask)
cv2.imshow('Hough Circles',color_img)
cv2.imshow('inpainted2',inpainted2)


cv2.waitKey(0)
cv2.destroyAllWindows()