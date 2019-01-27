import cv2
import numpy as np
import matplotlib as plt
import FourPointTransform

class ScoutingFormData:
    def __init__(self):
        self.team1 = []
        self.team10 = []
        self.team100 = []
        self.team1000 = []
        self.match1 = []
        self.match10 = []
        self.match100 = []
        self.color = []
        self.habCross = []
        self.hatchLow = []
        self.cargoLow = []
        self.cargoHigh = []
        self.habClimb = []
        self.foul = []
        self.card = []
        self.disabled = []
        self.playedDefense = []
        self.defenseAgainst = []

def ResizeImg(img, heightDesired):
    height, width, channels = img.shape

    ratio = width / height
    widthDesired = heightDesired * ratio

    img = cv2.resize(img, (int(widthDesired), int(heightDesired)))
    return img

def FitToQuestionBox(img):
    height = 600
    img = ResizeImg(img, height)

    # find largest contour
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 0)
    imgEdge = cv2.Canny(imgBlur, 75, 200)

    contours, hierarchy = cv2.findContours(imgEdge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

    # check contour is rectangle
    vertices = 0
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        tempVertices = cv2.approxPolyDP(c, 0.02 * perimeter, True)

        if len(tempVertices) == 4:
            vertices = tempVertices
            break
    # need error handling here

    imgBoxHighlight = img.copy()
    cv2.drawContours(imgBoxHighlight, [vertices], -1, (0, 255, 0), 3)

    # transform and crop to question box
    imgBox = FourPointTransform.FourPointTransform(img, vertices.reshape(4, 2))
    imgBox = ResizeImg(imgBox, height)

    # rotate from landscape to portrait
    if imgBox.shape[0] < imgBox.shape[1]:
        imgBoxHighlight = cv2.rotate(imgBoxHighlight, cv2.ROTATE_90_CLOCKWISE)
        imgBoxHighlight = ResizeImg(imgBoxHighlight, imgBoxHighlight.shape[1])

        imgBox = cv2.rotate(imgBox, cv2.ROTATE_90_CLOCKWISE)
        imgBox = ResizeImg(imgBox, imgBox.shape[1])

    # look for question box top
    grayThresh = 150
    imgThresh = cv2.threshold(imgBox, grayThresh, 255, cv2.THRESH_BINARY_INV)[1]
    heightPercentROI = 0.05

    cumSumTop = 0
    for i in range(0, imgThresh.shape[1]):
        for j in range(0, int(imgThresh.shape[0] * heightPercentROI)):
            cumSumTop += imgThresh[j, i, 1]
    cumSumTop /= (imgThresh.shape[1] * int(imgThresh.shape[0] * heightPercentROI * 255))

    cumSumBottom = 0
    for i in range(0, imgThresh.shape[1]):
        for j in range(int(imgThresh.shape[0]  * (1 - heightPercentROI)), imgThresh.shape[0] ):
            cumSumBottom += imgThresh[j, i, 1]
    cumSumBottom /= (imgThresh.shape[1] * int(imgThresh.shape[0]  * heightPercentROI * 255))

    # rotate to top
    if cumSumTop < cumSumBottom:
        imgBoxHighlight = cv2.rotate(imgBoxHighlight, cv2.ROTATE_180)
        imgBox = cv2.rotate(imgBox, cv2.ROTATE_180)

    cv2.imshow("imgBoxHighlight", imgBoxHighlight)
    return imgBox


def FindBubbles(imgBox):
    # fill in bubbles
    imgBoxGray = cv2.cvtColor(imgBox, cv2.COLOR_BGR2GRAY)
    imgBoxBlur = cv2.GaussianBlur(imgBoxGray, (5, 5), 0)
    imgBoxEdge = cv2.Canny(imgBoxBlur, 75, 200)

    contours, hierarchy = cv2.findContours(imgBoxEdge, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    radThreshMin = 5
    radThreshMax = 20
    imgBoxGrayFill = imgBoxGray.copy()
    for c in contours:
        (x, y), rad = cv2.minEnclosingCircle(c)

        if rad < radThreshMax and rad > radThreshMin:
            cv2.fillPoly(imgBoxGrayFill, pts=[c], color=(0, 0, 0))

    # count bubbles
    grayThresh = 150
    imgBoxThresh = cv2.threshold(imgBoxGrayFill, grayThresh, 255, cv2.THRESH_BINARY_INV)[1]
    contours, hierarchy = cv2.findContours(imgBoxThresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    bubbleContours = []
    bubbleCount = 0
    for c in contours:
        (x, y), rad = cv2.minEnclosingCircle(c)

        if rad < radThreshMax and rad > radThreshMin:
            bubbleContours.append(c)
            bubbleCount += 1

    if bubbleCount != 132:
        print(bubbleCount)
        print("Error incorrect bubble count")

    imgBubbleHighlight = imgBox.copy()
    cv2.drawContours(imgBubbleHighlight, bubbleContours, -1, (0, 0, 255), 3)

    cv2.imshow("imgBubbleHighlight", imgBubbleHighlight)

    return bubbleContours


def ReadScoutingFormData(imgBox, bubbleContours):
    bubbleY = []
    for c in bubbleContours:
        (x, y), rad = cv2.minEnclosingCircle(c)
        bubbleY.append(y)
    bubbleContours = [x for (y, x) in sorted(zip(bubbleY, bubbleContours), key=lambda pair: pair[0])]

    # find number of bubbles in each row
    heightDiffThresh = 10
    bubbleMatrix = []
    bubbleCount = 0
    (x, yOld), rad = cv2.minEnclosingCircle(bubbleContours[0])
    for c in bubbleContours:
        (x, y), rad = cv2.minEnclosingCircle(c)
        if (y - yOld) > heightDiffThresh:
            bubbleMatrix.append(bubbleCount)
            bubbleCount = 0
        bubbleCount += 1
        yOld = y
    bubbleMatrix.append(bubbleCount)

    if len(bubbleMatrix) != 19:
        print("Error incorrect row count")

    # find row values
    grayThresh = 150
    imgBoxThresh = cv2.threshold(imgBox, grayThresh, 255, cv2.THRESH_BINARY_INV)[1]
    imgBoxThresh = cv2.cvtColor(imgBoxThresh, cv2.COLOR_BGR2GRAY)

    bubbleFillThresh = 200
    bubbleMatrix2 = []
    totalBubbleCount = 0
    for c2 in bubbleMatrix:
        bubbleCount = 0
        rowValue = []

        bubbleX = []
        rowBubbleContours = bubbleContours[totalBubbleCount:totalBubbleCount + c2]
        for c in rowBubbleContours:
            (x, y), rad = cv2.minEnclosingCircle(c)
            bubbleX.append(x)
        rowBubbleContours = [x for (y, x) in sorted(zip(bubbleX, rowBubbleContours), key=lambda pair: pair[0])]

        for c in rowBubbleContours:
            bubbleCount += 1

            mask = np.zeros(imgBoxThresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask = cv2.bitwise_and(imgBoxThresh, imgBoxThresh, mask=mask)
            fillPixels = cv2.countNonZero(mask)

            if fillPixels > bubbleFillThresh:
                rowValue = bubbleCount

        bubbleMatrix2.append(rowValue)
        totalBubbleCount += c2
    print(bubbleMatrix2)


if __name__== "__main__":
    img = cv2.imread("C:/frc-2019-scout/Filled Out 2481 Scouting Form 2019.jpeg", cv2.IMREAD_COLOR)
    if img is None:
        print("Error failed to read image")

    imgBox = FitToQuestionBox(img)
    bubbleContours = FindBubbles(imgBox)
    ReadScoutingFormData(imgBox, bubbleContours)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



#
# for c2 in qMatrix:
#     qCnt = 0
#     qValue = []
#     rowQContours = sorted(qContours[c2 - 1:c2], key=lambda ctr: cv2.boundingRect(ctr)[0])
#     print(c2 - 1, c2)
#     for c in rowQContours:
#         qCnt += 1
#         mask = np.zeros(imgThresh.shape, dtype="uint8")
#         cv2.drawContours(mask, [c], -1, 255, -1)
#
#         mask = cv2.bitwise_and(imgThresh, imgThresh, mask=mask)
#         total = cv2.countNonZero(mask)
#         if total > qFillThresh:
#             qValue = qCnt
#     qMatrix2.append(qValue)
#
# print(qMatrix2)
#
# imgThresh = cv2.cvtColor(imgThresh, cv2.COLOR_GRAY2RGB)
# cv2.drawContours(imgThresh, qContours, -1, (0, 255, 0), 1)
#
# # cv2.imshow("img", img)
# # cv2.imshow("imgThresh", imgThresh)

