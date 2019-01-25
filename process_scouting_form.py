import cv2
import numpy as np
import matplotlib as plt
import four_point_transform

class formData:
    def __init__(self):
        self.team1 = []
        self.team10 = []
        self.team100 = []
        self.team1000 = []
        self.match1 = []
        self.match10 = []
        self.match100 = []
        self.color = []
        self.habCross = 0
        self.hatchLow = 0
        self.cargoLow = 0
        self.cargoHigh = 0
        self.habClimb = 0
        self.foul = 0
        self.card = 0
        self.disabled = 0
        self.playedDefense = 0
        self.defenseAgainst = 0

img = cv2.imread("C:/frc-2019-scout/Filled Out 2481 Scouting Form 2019.jpeg", cv2.IMREAD_COLOR)

heightDesired = 800
height, width, channels = img.shape
ratio = width / height
widthDesired = heightDesired * ratio
img = cv2.resize(img, (int(widthDesired), int(heightDesired)))

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 0)
imgEdge = cv2.Canny(imgBlur, 75, 200)

contours, hierarchy = cv2.findContours(imgEdge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

vertices = 0
for c in contours:
    perimeter = cv2.arcLength(c, True)
    tempVertices = cv2.approxPolyDP(c, 0.02 * perimeter, True)

    if len(tempVertices) == 4:
        vertices = tempVertices
        break
# cv2.drawContours(img, [vertices], -1, (0, 255, 0), 5)

imgWarped = four_point_transform.four_point_transform(img, vertices.reshape(4, 2))
if imgWarped.shape[0] < imgWarped.shape[1]:
    imgWarped = cv2.rotate(imgWarped, cv2.ROTATE_90_CLOCKWISE)

    height, width, channels = imgWarped.shape
    ratio = width / height
    widthDesired = heightDesired * ratio
    imgWarped = cv2.resize(imgWarped, (int(widthDesired), int(heightDesired)))

grayThresh = 150
imgThresh = cv2.threshold(imgWarped, grayThresh, 255, cv2.THRESH_BINARY_INV)[1]

heightPercentROI = 0.05
cumSumTop = 0
for i in range(0, int(widthDesired)):
    for j in range(0, int(heightDesired * heightPercentROI)):
        cumSumTop += imgThresh[j, i, 1]
cumSumTop /= (int(widthDesired) * int(heightDesired * heightPercentROI * 255))

cumSumBottom = 0
for i in range(0, int(widthDesired)):
    for j in range(int(heightDesired * (1 - heightPercentROI)), int(heightDesired)):
        cumSumBottom += imgThresh[j, i, 1]
cumSumBottom /= (int(widthDesired) * int(heightDesired * heightPercentROI * 255))

if cumSumTop < cumSumBottom:
    imgThresh = cv2.rotate(imgThresh, cv2.ROTATE_180)

imgThresh = cv2.cvtColor(imgThresh, cv2.COLOR_BGR2GRAY)
imgThreshFill = imgThresh.copy()
contours, hierarchy = cv2.findContours(imgThreshFill, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

radThreshMax = 100
for c in contours:
    (x, y), radius = cv2.minEnclosingCircle(c)

    if radius < radThreshMax:
        cv2.fillPoly(imgThreshFill, pts=[c], color=(255, 255, 255))

contours, hierarchy = cv2.findContours(imgThreshFill, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

qContours = []
qCnt = 0
for c in contours:
    (x, y), radius = cv2.minEnclosingCircle(c)

    if radius < radThreshMax:
        qContours.append(c)
        qCnt += 1

qContours = sorted(qContours, key=lambda ctr: cv2.boundingRect(ctr)[1])

heightDiffThresh = 5
qMatrix = []
qValue = []
qCnt = 0
(x, yOld), radius = cv2.minEnclosingCircle(qContours[0])
for c in qContours:
    (x, y), radius = cv2.minEnclosingCircle(c)
    if (y - yOld) > heightDiffThresh:
        qMatrix.append(qCnt)
        qValue = []
        qCnt = 0
    qCnt += 1
    yOld = y
qMatrix.append(qCnt)
print(qMatrix)

qMatrix2 = []
qFillThresh = 300
rowQContours = sorted(qContours[0:qMatrix[0]], key=lambda ctr: cv2.boundingRect(ctr)[0])
qCnt = 0
qValue = []
for c in rowQContours:
    qCnt += 1
    mask = np.zeros(imgThresh.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)

    mask = cv2.bitwise_and(imgThresh, imgThresh, mask=mask)
    total = cv2.countNonZero(mask)
    if total > qFillThresh:
        qValue = qCnt
qMatrix2.append(qValue)

for c2 in qMatrix:
    qCnt = 0
    qValue = []
    rowQContours = sorted(qContours[c2 - 1:c2], key=lambda ctr: cv2.boundingRect(ctr)[0])
    print(c2 - 1, c2)
    for c in rowQContours:
        qCnt += 1
        mask = np.zeros(imgThresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)

        mask = cv2.bitwise_and(imgThresh, imgThresh, mask=mask)
        total = cv2.countNonZero(mask)
        if total > qFillThresh:
            qValue = qCnt
    qMatrix2.append(qValue)

print(qMatrix2)

imgThresh = cv2.cvtColor(imgThresh, cv2.COLOR_GRAY2RGB)
cv2.drawContours(imgThresh, qContours, -1, (0, 255, 0), 1)

# cv2.imshow("img", img)
# cv2.imshow("imgThresh", imgThresh)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
