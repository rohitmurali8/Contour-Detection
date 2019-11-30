import numpy as np
import cv2
import math

pathName = "C:\Data\Software\ECE5554 FA19 HW3 images\VAoutline.png" 
MAXCONTOUR = 5000
doLogging = False

def showImage(img, name):
    cv2.imshow(name, img)

def saveImage(img, name):
    cv2.imwrite(pathName + name + ".png", img)

def GaussArea(pts):

    n = len(pts)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1]
    area = abs(area) / 2.0
    
    return area

def length(contour, i):
    size = contour.shape[0]
    if i == size - 1:
        l = math.sqrt((contour[i,0] - contour[0,0])**2 + (contour[i,1] - contour[0,1])**2)
    else:
        l = math.sqrt((contour[i,0] - contour[i-1,0])**2 + (contour[i,1] - contour[i - 1,1])**2)
    return l

def theta(contour, i):
    size = contour.shape[0]
    if i == size - 1:
        theta = abs(math.atan((contour[i,0] - contour[i-1,0])/(contour[i,1] - contour[i-1,1])) - math.atan((contour[0,0] - contour[i,0])/(contour[0,1] - contour[i,1])))
    else:
        theta = abs(math.atan((contour[i,0] - contour[i-1,0])/(contour[i,1] - contour[i-1,1])) - math.atan((contour[i+1,0] - contour[i,0])/(contour[i+1,1] - contour[i,1])))
    return theta

def onePassDCE(contour):
    size = contour.shape[0]
    rele = []
    for i in range(size):
        if i < size - 1:
            rel = (theta(contour, i)*length(contour,i)*length(contour,i+1))/(length(contour,i) + length(contour,i+1))
        if i == size - 1:
            rel = (theta(contour, i)*length(contour,i)*length(contour,0))/(length(contour,i) + length(contour,0))
        rele.append(rel)
        
    rele = np.asarray(rele)
    rele = np.reshape(rele,(size,1)) 
    a = min(rele)
    index = np.argmin(rele)
    trimmed_contour = np.delete(contour,idx,0)
    return trimmed_contour

def getPoint(direction, a):
    if direction == "up":
        d = [(a[0] - 1,a[1] - 1),(a[0] - 1,a[1]),(a[0] - 1,a[1] + 1)]
    elif direction == "down":
        d = [(a[0] + 1,a[1] + 1),(a[0] + 1,a[1]),(a[0] + 1,a[1] - 1)]
    elif direction == "left":
        d = [(a[0] + 1,a[1] - 1),(a[0] ,a[1] - 1),(a[0] - 1,a[1] - 1)]
    elif direction == "right":
        d = [(a[0] - 1,a[1] + 1),(a[0] ,a[1] + 1),(a[0] + 1,a[1] + 1)]
    return d

def getNextDirection(current_direction, i):
    if i == 0:
        if current_direction == "up":
            nextDirection = "left"
        elif current_direction == "down":
            nextDirection = "right"
        elif current_direction == "left":
            nextDirection = "down"
        elif current_direction == "right":
            nextDirection = "up"
    else:
        nextDirection = current_direction
    return nextDirection

def getNewDirection(current_direction):
    if current_direction == "up":
        nextDirection = "right"
    elif current_direction == "down":
        nextDirection = "left"
    elif current_direction == "left":
        nextDirection = "up"
    elif current_direction == "right":
        nextDirection = "down"
    return nextDirection

def Pavlidis(img, start):
    contour_point = []
    count = 0
    c = 0
    direction = "up"
    point = start
    while True:
        front_pixel = getPoint(direction,point)

        for i in range(len(front_pixel)):
            if binary[front_pixel[i]] > 0:
                contour_point.append(front_pixel[i])
                count = count + 1
                point = front_pixel[i]
                direction = getNextDirection(direction, i)
                break
        if count == 0:
            direction = getNewDirection(direction)
        if point == start:
            break
        count = 0
    print(len(contour_point))
    return contour_point

def showContour(ctr, img, name):
    contourImage = img
    length = ctr.shape[0]
    row, col = img.shape
    contour = np.zeros((row,col))
    for count in range(length):
        contourImage[ctr[count, 0], ctr[count, 1]] = 255
        contour[ctr[count, 0], ctr[count, 1]] = 255
        cv2.line(contour,(ctr[count, 1], ctr[count, 0]), \
                 (ctr[(count+1)%length, 1], ctr[(count+1)%length, 0]),(255,255,255),1)
        cv2.line(contourImage,(ctr[count, 1], ctr[count, 0]), \
                 (ctr[(count+1)%length, 1], ctr[(count+1)%length, 0]),(255,255,255),1)
    showImage(contourImage, name)
    showImage(contour, "Detected")
    saveImage(contourImage, name)

inputImage = cv2.imread(pathName, cv2.IMREAD_GRAYSCALE)
thresh = 70
ret, binary = cv2.threshold(inputImage, thresh, 255, cv2.THRESH_BINARY)
(height, width) = binary.shape
ystt = np.uint8(height/2)
for xstt in range(width):
    if (binary[ystt, xstt] > 0):
        break
start = (ystt, xstt)
contour = Pavlidis(binary, start)
contour = np.asarray(contour)
showContour(contour, inputImage, "CONTOUR")
print(contour, GaussArea(contour))

for step in range(6):
    numLoops = math.floor(contour.shape[0]/2)
    for idx in range(numLoops):
        contour = onePassDCE(contour)
        showContour(contour, np.zeros_like(inputImage), "STEP"+str(step))
        print(numLoops, contour.shape, GaussArea(contour))
cv2.waitKey(0)
