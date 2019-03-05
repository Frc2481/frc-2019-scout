import cv2
import numpy as np
import matplotlib as plt
import FourPointTransform
import os
import csv
import copy
import sys

showImages = True

class ScoutingFormData:
    def __init__(self):
        self.team = ""
        self.match = ""
        self.color = ""
        self.habCross = ""
        self.hatchLowSandstorm = ""
        self.hatchHighSandstorm = ""
        self.cargoLowSandstorm = ""
        self.cargoHighSandstorm = ""
        self.hatchLowTeleop = ""
        self.hatchHighTeleop = ""
        self.cargoLowTeleop = ""
        self.cargoHighTeleop = ""
        self.habClimb = ""
        self.foul = ""
        self.card = ""
        self.disabled = ""
        self.playedDefense = ""
        self.defenseAgainst = ""


def FormatBlankData(data):
    if not data:
        data = 0

    return data


def ResizeImg(img, heightDesired):
    height, width, channels = img.shape

    ratio = width / height
    widthDesired = heightDesired * ratio

    img = cv2.resize(img, (int(widthDesired), int(heightDesired)))
    return img, False


def FitToQuestionBox(img):
    height = 1200
    img, isError = ResizeImg(img, height)
    if isError:
        print("\033[91m" + "Error image resize failed" + "\033[0m")
        return [], True

    # find largest contour
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgEdge = cv2.Canny(img, 75, 200)

    contours, hierarchy = cv2.findContours(imgEdge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

    # check contour is rectangle
    vertices = []
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        tempVertices = cv2.approxPolyDP(c, 0.02 * perimeter, True)

        if len(tempVertices) == 4:
            vertices = tempVertices
            break

    if len(vertices) != 4:
        print("\033[91m" + "Error question box not found" + "\033[0m")
        return [], True

    imgBoxHighlight = img.copy()
    cv2.drawContours(imgBoxHighlight, [vertices], -1, (0, 255, 0), 3)

    # transform and crop to question box
    imgBox = FourPointTransform.FourPointTransform(img, vertices.reshape(4, 2))
    imgBox, tempIsError = ResizeImg(imgBox, height)
    isError = isError and tempIsError

    # rotate from landscape to portrait
    if imgBox.shape[0] < imgBox.shape[1]:
        imgBoxHighlight = cv2.rotate(imgBoxHighlight, cv2.ROTATE_90_CLOCKWISE)
        imgBoxHighlight, tempIsError = ResizeImg(imgBoxHighlight, imgBoxHighlight.shape[1])
        isError = isError and tempIsError

        imgBox = cv2.rotate(imgBox, cv2.ROTATE_90_CLOCKWISE)
        imgBox, isError = ResizeImg(imgBox, imgBox.shape[1])
        isError = isError and tempIsError

    # look for question box top
    imgGray = cv2.cvtColor(imgBox, cv2.COLOR_BGR2GRAY)
    imgThresh = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    heightPercentROI = 0.05

    cumSumTop = 0
    for i in range(0, imgThresh.shape[1]):
        for j in range(0, int(imgThresh.shape[0] * heightPercentROI)):
            cumSumTop += imgThresh[j, i]
    cumSumTop /= (imgThresh.shape[1] * int(imgThresh.shape[0] * heightPercentROI * 255))

    cumSumBottom = 0
    for i in range(0, imgThresh.shape[1]):
        for j in range(int(imgThresh.shape[0]  * (1 - heightPercentROI)), imgThresh.shape[0] ):
            cumSumBottom += imgThresh[j, i]
    cumSumBottom /= (imgThresh.shape[1] * int(imgThresh.shape[0]  * heightPercentROI * 255))

    # rotate to top
    if cumSumTop < cumSumBottom:
        imgThresh = cv2.rotate(imgThresh, cv2.ROTATE_180)
        imgBoxHighlight = cv2.rotate(imgBoxHighlight, cv2.ROTATE_180)
        imgBox = cv2.rotate(imgBox, cv2.ROTATE_180)

    if showImages:
        cv2.imshow("imgThresh", imgThresh)
        cv2.imshow("imgBoxHighlight", imgBoxHighlight)

    return imgBox, False


def FindBubbles(imgBox):
    # fill in bubbles
    imgGray = cv2.cvtColor(imgBox, cv2.COLOR_BGR2GRAY)
    imgEdge = cv2.Canny(imgBox, 75, 200)

    contours, hierarchy = cv2.findContours(imgEdge, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    radThreshMin = 5
    radThreshMax = 25
    imgGrayFill = imgGray.copy()
    for c in contours:
        (x, y), rad = cv2.minEnclosingCircle(c)

        if rad < radThreshMax and rad > radThreshMin:
            imgGrayFill = cv2.circle(imgGrayFill, (int(x), int(y)), int(rad), color=(0, 0, 0), thickness=-1, lineType=8, shift=0)

    # count bubbles
    imgThreshFill = cv2.threshold(imgGrayFill, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    contours, hierarchy = cv2.findContours(imgThreshFill, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    bubbleContours = []
    bubbleCount = 0
    for c in contours:
        (x, y), rad = cv2.minEnclosingCircle(c)

        if rad < radThreshMax and rad > radThreshMin:
            bubbleContours.append(c)
            bubbleCount += 1

    expectedBubbleCount = 148
    if bubbleCount != expectedBubbleCount:
        print(bubbleCount)
        print("\033[91m" + "Error incorrect bubble count" + "\033[0m")
        return [], True

    imgBubbleHighlight = imgBox.copy()
    cv2.drawContours(imgBubbleHighlight, bubbleContours, -1, (0, 0, 255), 3)

    if showImages:
        cv2.imshow("imgGrayFill", imgGrayFill)
        cv2.imshow("imgBubbleHighlight", imgBubbleHighlight)

    return bubbleContours, False


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

    expectedBubbleMatrix = [10, 10, 10, 10, 10, 10, 10, 2, 2, 4, 4, 4, 4, 12, 8, 12, 8, 3, 10, 2, 1, 1, 1]
    if bubbleMatrix != expectedBubbleMatrix:
        print("\033[91m" + "Error incorrect bubble matrix" + "\033[0m")
        return [], True

    # find row values
    imgGray = cv2.cvtColor(imgBox, cv2.COLOR_BGR2GRAY)
    imgThresh = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    bubbleFillThreshPercent = 0.7
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
            
            area = cv2.contourArea(c)

            mask = np.zeros(imgThresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask = cv2.bitwise_and(imgThresh, imgThresh, mask=mask)
            fillPixels = cv2.countNonZero(mask)
            percentFill = (fillPixels / area)

            if percentFill > bubbleFillThreshPercent:
                rowValue = bubbleCount

        bubbleMatrix2.append(rowValue)
        totalBubbleCount += c2

    # convert bubble data to form data
    scoutingFormData = ScoutingFormData()
    if bubbleMatrix2[0] and bubbleMatrix2[1] and bubbleMatrix2[2] and bubbleMatrix2[3]:
        scoutingFormData.team = \
            (bubbleMatrix2[0] - 1) * 1000 \
            + (bubbleMatrix2[1] - 1) * 100 \
            + (bubbleMatrix2[2] - 1) * 10 \
            + (bubbleMatrix2[3] - 1)
    else:
        print("\033[91m" + "Error team not defined" + "\033[0m")
        return [], True

    if bubbleMatrix2[4] and bubbleMatrix2[5] and bubbleMatrix2[6]:
        scoutingFormData.match = \
            (bubbleMatrix2[4] - 1) * 100 \
            + (bubbleMatrix2[5] - 1) * 10 \
            + (bubbleMatrix2[6] - 1)
    else:
        print("\033[91m" + "Error match not defined")
        return [], True

    if bubbleMatrix2[7]:
        scoutingFormData.color = bubbleMatrix2[7] - 1
    else:
        print("\033[91m" + "Error color not defined" + "\033[0m")
        return [], True

    scoutingFormData.habCross = FormatBlankData(bubbleMatrix2[8])
    scoutingFormData.hatchLowSandstorm = FormatBlankData(bubbleMatrix2[9])
    scoutingFormData.hatchHighSandstorm = FormatBlankData(bubbleMatrix2[10])
    scoutingFormData.cargoLowSandstorm = FormatBlankData(bubbleMatrix2[11])
    scoutingFormData.cargoHighSandstorm = FormatBlankData(bubbleMatrix2[12])
    scoutingFormData.hatchLowTeleop = FormatBlankData(bubbleMatrix2[13])
    scoutingFormData.hatchHighTeleop = FormatBlankData(bubbleMatrix2[14])
    scoutingFormData.cargoLowTeleop = FormatBlankData(bubbleMatrix2[15])
    scoutingFormData.cargoHighTeleop = FormatBlankData(bubbleMatrix2[16])
    scoutingFormData.habClimb = FormatBlankData(bubbleMatrix2[17])
    scoutingFormData.foul = FormatBlankData(bubbleMatrix2[18])
    scoutingFormData.card = FormatBlankData(bubbleMatrix2[19])
    scoutingFormData.disabled = FormatBlankData(bubbleMatrix2[20])
    scoutingFormData.playedDefense = FormatBlankData(bubbleMatrix2[21])
    scoutingFormData.defenseAgainst = FormatBlankData(bubbleMatrix2[22])

    return scoutingFormData, False


def CreateOutputFileFromMatchSchedule(matchScheduleFilepath, outputFilepath):
    # read match schedule
    if not os.path.isfile(matchScheduleFilepath):
        print("\033[91m" + "Error failed to read match schedule" + "\033[0m")
        return
    
    # create output file if haven't already
    if os.path.isfile(outputFilepath):
        return
    
    print()
    print("\033[95m" + "Processing match schedule..." + "\033[0m")
    
    match = ScoutingFormData()
    matchList = []
    
    with open(matchScheduleFilepath, 'r',  newline="") as csvFile:
        csvReader = csv.reader(csvFile, delimiter=",")

        # skip headers
        next(csvReader)

        # loop through rows and read matches
        for row in csvReader:
            match.match = int(row[0].replace("Qualification ",""))
            match.team = int(row[2])
            match.color = 0 # red
            matchList.append(copy.deepcopy(match))
            
            match.team = int(row[3])
            match.color = 0 # red
            matchList.append(copy.deepcopy(match))
            
            match.team = int(row[4])
            match.color = 0 # red
            matchList.append(copy.deepcopy(match))
            
            match.team = int(row[5])
            match.color = 1 # blue
            matchList.append(copy.deepcopy(match))
            
            match.team = int(row[6])
            match.color = 1 # blue
            matchList.append(copy.deepcopy(match))
            
            match.team = int(row[7])
            match.color = 1 # blue
            matchList.append(copy.deepcopy(match))

    with open(outputFilepath, "w",  newline="") as csvFile:
        csvWriter = csv.writer(csvFile)
        
        # write headers
        csvWriter.writerow([
            "Match",
            "Team",
            "Color",
            "HAB Cross",
            "Hatch Low Sandstorm",
            "Hatch High Sandstorm",
            "Cargo Low Sandstorm",
            "Cargo High Sandstorm",
            "Hatch Low Teleop",
            "Hatch High Teleop",
            "Cargo Low Teleop",
            "Cargo High Teleop",
            "HAB Climb",
            "Foul",
            "Card",
            "Disabled",
            "Played Defense",
            "Defense Against",
        ])
        
        # write matches
        for match in matchList:
            csvWriter.writerow([
                match.match,
                match.team,
                match.color,
                match.habCross,
                match.hatchLowSandstorm,
                match.hatchHighSandstorm,
                match.cargoLowSandstorm,
                match.cargoHighSandstorm,
                match.hatchLowTeleop,
                match.hatchHighTeleop,
                match.cargoLowTeleop,
                match.cargoHighTeleop,
                match.habClimb,
                match.foul,
                match.card,
                match.disabled,
                match.playedDefense,
                match.defenseAgainst,
            ])

    print("\033[92m" + "Processed match schedule" + "\033[0m")
    return


def WriteScoutingFormDataToOutputFile(scoutingFormData, outputFilepath):
    with open(outputFilepath, "r",  newline="") as csvFile:
        csvReader = csv.reader(csvFile, delimiter=",")
        tempData = []
        
        # loop through rows and find match
        matchFound = False
        rowCnt = 0
        for row in csvReader:
            # skip headers
            if not rowCnt == 0:
                # write form data to output file if match found
                if (int(row[0]) == scoutingFormData.match) and (int(row[1]) == scoutingFormData.team):
                    matchFound = True
                    tempRow = row
                    tempRowCnt = rowCnt
                    tempRow[0] = scoutingFormData.match
                    tempRow[1] = scoutingFormData.team
                    tempRow[2] = scoutingFormData.color
                    tempRow[3] = scoutingFormData.habCross
                    tempRow[4] = scoutingFormData.hatchLowSandstorm
                    tempRow[5] = scoutingFormData.hatchHighSandstorm
                    tempRow[6] = scoutingFormData.cargoLowSandstorm
                    tempRow[7] = scoutingFormData.cargoHighSandstorm
                    tempRow[8] = scoutingFormData.hatchLowTeleop
                    tempRow[9] = scoutingFormData.hatchHighTeleop
                    tempRow[10] = scoutingFormData.cargoLowTeleop
                    tempRow[11] = scoutingFormData.cargoHighTeleop
                    tempRow[12] = scoutingFormData.habClimb
                    tempRow[13] = scoutingFormData.foul
                    tempRow[14] = scoutingFormData.card
                    tempRow[15] = scoutingFormData.disabled
                    tempRow[16] = scoutingFormData.playedDefense
                    tempRow[17] = scoutingFormData.defenseAgainst

            tempData.append(row)
            rowCnt += 1

    if not matchFound:
        print("\033[91m" + "Error match not found in match schedule" + "\033[0m")
        return True

    with open(outputFilepath, "w",  newline="") as csvFile:
        csvWriter = csv.writer(csvFile)
        tempData[tempRowCnt] = tempRow
        csvWriter.writerows(tempData)

    return False


if __name__== "__main__":
    workDir = os.getcwd()

    # read match schedule and create output file
    matchScheduleFilename = "Match Schedule.csv"
    matchScheduleFilepath = os.path.join(workDir, matchScheduleFilename)
    
    outputFilename = "Raw Form Data.csv"
    outputFilepath = os.path.join(workDir, outputFilename)
    
    CreateOutputFileFromMatchSchedule(matchScheduleFilepath, outputFilepath)
    
    # loop through unprocessed images
    unprocessedDirName = os.path.join(workDir, "Unprocessed Forms")
    processedDirName = os.path.join(workDir, "Processed Forms")
    unprocessedDir = os.fsencode(unprocessedDirName)
    for file in os.listdir(unprocessedDir):
        unprocessedFilename = os.fsdecode(file)
        unprocessedFilepath = os.path.join(unprocessedDirName, unprocessedFilename)
        if unprocessedFilename.endswith(".jpg") or unprocessedFilename.endswith(".jpeg") or unprocessedFilename.endswith(".JPG"):
            print()
            print("\033[95m" + "Processing " + unprocessedFilename + "..." + "\033[0m")
            
            img = cv2.imread(unprocessedFilepath, cv2.IMREAD_COLOR)
            if img is None:
                print("\033[91m" + "Error failed to read image" + "\033[0m")
                continue

            imgBox, isError = FitToQuestionBox(img)
            if isError:
                continue

            bubbleContours, isError = FindBubbles(imgBox)
            if isError:
                continue

            scoutingFormData, isError = ReadScoutingFormData(imgBox, bubbleContours)
            if isError:
                continue
                
            isError = WriteScoutingFormDataToOutputFile(scoutingFormData, outputFilepath)
            if isError:
                continue
            
            # move image to processed
            processedFilename = str(scoutingFormData.match) + "_" + str(scoutingFormData.team) + ".jpg"
            processedFilepath = os.path.join(processedDirName, processedFilename)
            if os.path.isfile(processedFilepath):
                os.remove(processedFilepath)
            os.rename(unprocessedFilepath, processedFilepath)
            
            print("\033[92m" + "Processed " + processedFilename + "\033[0m")
            
        else:
            continue
            
        sys.stdout.flush()

    if showImages:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
