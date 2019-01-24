import cv2
import imutils
import numpy as np
import four_point_transform

img = cv2.imread("C:/frc-2019-scout/Filled Out 2481 Scouting Form 2019.jpg", cv2.IMREAD_COLOR)

height = 800
img = imutils.resize(img, height=height)

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 0)
imgEdge = cv2.Canny(imgBlur, 75, 200)

contours = cv2.findContours(imgEdge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]

vertices = 0
for c in contours:
    perimeter = cv2.arcLength(c, True)
    tempVertices = cv2.approxPolyDP(c, 0.02 * perimeter, True)

    if len(tempVertices) == 4:
        vertices = tempVertices
        break
cv2.drawContours(img, [vertices], -1, (0, 255, 0), 5)

imgWarped = four_point_transform.four_point_transform(img, vertices.reshape(4, 2))
if imgWarped.shape[0] < imgWarped.shape[1]:
    imgWarped = imutils.rotate(imgWarped, -90)

imgWarpedGray = cv2.cvtColor(imgWarped, cv2.COLOR_BGR2GRAY)
imgThresh = cv2.threshold(imgWarpedGray, 180, 255, cv2.THRESH_BINARY_INV)[1]

cnts = cv2.findContours(imgThresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cnts = imutils.grab_contours(cnts)
questionCnts = []

# loop over the contours
for c in cnts:
    # compute the bounding box of the contour, then use the
    # bounding box to derive the aspect ratio
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)

    (x2, y2), radius = cv2.minEnclosingCircle(c)
    area = cv2.contourArea(c)

    # in order to label the contour as a question, region
    # should be sufficiently wide, sufficiently tall, and
    # have an aspect ratio approximately equal to 1
    if w<50 and h<50 and w>10 and h>10 and area > 3.14 * radius * radius * 0.5:
        questionCnts.append(c)
        print(area, 3.14 * radius * radius)

cv2.drawContours(imgWarped, questionCnts, -1, (0, 0, 255), 1)

cv2.imshow("img", imgWarped)
cv2.waitKey(0)
cv2.destroyAllWindows()
