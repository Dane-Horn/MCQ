import numpy as np
import cv2
from math import atan2
import math
import os


# from sys import argv
def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1)
    result = cv2.warpAffine(
        image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


# draw smallest minimum enclosing circles around contours
def drawMinEnclose(resized, circles):
    (x, y), radius = cv2.minEnclosingCircle(circles)
    center = (int(x), int(y))
    radius = int(radius)
    cv2.circle(resized, center, radius, (0, 255, 0), 2)


# get corners of inner mcq sheet
def getCorners(image, drawImage=None):  # TODO
    def intCircle(circle):
        return (int(circle[0]), int(circle[1]))

    gray = image.copy()
    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)

    # close to get rid of as many white spots in the middle of a circle as possible
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
                               kernel, iterations=9)
    # erode to make black shapes as large as possible
    eroding = cv2.morphologyEx(closing, cv2.MORPH_ERODE,
                               kernel, iterations=15)

    contours, _ = cv2.findContours(
        eroding, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    circleArr = []
    for circles in contours:
        area = cv2.contourArea(circles)
        # filter contours
        if area < 200 or area > 50000:
            continue

        if len(circles) < 5:
            continue

        (x, y), _ = cv2.minEnclosingCircle(circles)
        # draw circles on drawImage if arg is set
        if type(drawImage) != None:
            drawMinEnclose(drawImage, circles)
        circleArr.append((x, y))

    # sort and extract corners based on extreme values
    # top left - lowest x, y combination
    # top right - highest x with lowest y combination
    # bottom left - lowest x with highest y combination
    # bottom right - highest x y combination
    xSorted = sorted(circleArr, key=lambda x: x[0])

    leftCircles = sorted(xSorted[:2], key=lambda x: x[1])
    topLeft = intCircle(leftCircles[0])
    bottomLeft = intCircle(leftCircles[1])
    rightCircles = sorted(xSorted[-2:], key=lambda x: x[1])
    topRight = intCircle(rightCircles[0])
    bottomRight = intCircle(rightCircles[1])
    return (topLeft, bottomLeft, topRight, bottomRight)


# crops out mcq sheet based on corners found
def getImportant(image, corners, expand=0):
    topLeft, bottomLeft, topRight, bottomRight = corners
    left_right = math.sqrt(
        (bottomRight[0]-topLeft[0])**2 + (bottomRight[1]-topLeft[1])**2)
    right_left = math.sqrt(
        (topRight[0]-bottomLeft[0])**2 + (bottomLeft[1]-topRight[1])**2)
    x1, y1, x2, y2 = 0, 0, 0, 0
    if left_right > right_left:
        ((x1, y1), (x2, y2)) = (topLeft, bottomRight)
    else:
        ((x1, y1), (x2, y2)) = (topRight, bottomLeft)
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    minx = min(x1, x2)
    maxx = max(x1, x2)
    miny = min(y1, y2)
    maxy = max(y1, y2)
    img = image[(miny-expand):(maxy+expand), (minx-expand):(maxx+expand)]
    return img


# corrects angle using the angle between two points; either the top left and top right corners
# or the bottom left and bottom right corners
def correctAngle(img, p1, p2):
    angle = atan2(p2[1]-p1[1], p2[0]-p1[0])
    angle = math.degrees(angle)
    if angle < 0:
        angle = 360 + angle
    rotated = rotateImage(img, angle)
    return rotated


def check180(img):
    rows, cols = img.shape
    lowLeft = img[int(rows / 1.025):int(rows / 1.01), :int(cols / 6)].copy()
    roi = img[int(rows / 1.04):, :int(cols / 19.5)]
    _, roi = cv2.threshold(roi, 255, 255, cv2.THRESH_BINARY)
    img[int(rows / 1.04):, :int(cols / 19.5)] = roi
    _, lowLeft = cv2.threshold(
        lowLeft, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    lowLeft = cv2.morphologyEx(lowLeft, cv2.MORPH_ERODE,
                               kernel, iterations=3)
    rows, cols = lowLeft.shape
    nPixels = rows * cols
    if cv2.countNonZero(lowLeft) > nPixels * 0.80:
        img = rotateImage(img, 180)
    cv2.imshow('lowleft', lowLeft)
    cv2.imshow('i', cv2.resize(img, (500, 500)))
    cv2.waitKey(0)
    return img


def cropAndCorrect(img, crop=False):
    sheet = img
    if crop:
        sheet = img[1050:, :]
    gray = cv2.cvtColor(sheet, cv2.COLOR_BGR2GRAY)
    corners = getCorners(gray)
    rotated = correctAngle(gray, corners[0], corners[2])
    cropped = getImportant(rotated, corners, expand=-90)
    cropped = check180(cropped)
    # corners = getCorners(cropped)
    return cropped


# get question part of inner sheet
def multipleChoiceRegion(img):
    rows, cols = img.shape
    return img[:int(rows/1.028), int(cols/3.2):].copy()


# get student number part of inner sheet
def studentNumberRegion(img):
    rows, cols = img.shape
    return img[int(rows / 7.5):int(rows / 1.17), int(cols / 29): int(cols / 3.45)].copy()


# get task number part from student number part
def taskFromStudentNumber(stNumber):
    rows, cols = stNumber.shape
    return stNumber[int(rows / 2.1):int(rows / 1.16), int(cols / 1.45):int(cols / 1.05)].copy()


# clean the task number from the student number region, just to keep it clean
def removeTaskFromStudentNumber(stNumber):
    ret = stNumber.copy()
    rows, cols = stNumber.shape
    ret[int(rows / 2.4):, int(cols / 1.7):] = 255
    return ret


# general method to get the nth (0 to max-1) circle filled in given the circles to look at
# optional arguments:
# max - the number of circles in the picture
# display - display partial processing as images are generated
# horizontal - changes the number to pass through column-wise rather than the default row-wise
# multiple - return an array of values found instead of the last value found
def getNumber(img, max=10, display=False, horizontal=False, multiple=False):
    _, thresh = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
                               kernel, iterations=5)

    rows, cols = img.shape
    if not horizontal:
        circles = [closing[int(rows * ((x - 1) / max)):int(rows * (x / max)), :].copy()
                   for x in range(1, max+1)]
    else:
        circles = [closing[:, int(cols * ((x - 1) / max)):int(cols * (x / max))].copy()
                   for x in range(1, max+1)]
    number = 0
    found = False
    if multiple:
        answers = []
    for i, circle in enumerate(circles):
        c = cv2.resize(circle, (100, 100))
        nPixels = 100*100
        check = cv2.bitwise_not(c)
        if (cv2.countNonZero(check) >= nPixels*0.3):
            found = True
            if multiple:
                answers.append(i)
            else:
                number = i
        if display:
            cv2.imshow(f'c{i}', c)
            cv2.moveWindow(f'c{i}', 120, 100*i)
    if display:
        cv2.waitKey(0)
    if found:
        if multiple:
            return answers
        return number
    else:
        if multiple:
            return []
        return -1


def numberToLetter(n):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    if n < 0:
        return '?'
    return alphabet[n]


def outNumber(n):
    if n < 0:
        return '?'
    return f'{n}'


def outList(arr):
    if arr == []:
        return '?'
    else:
        arr = list(map(lambda item: numberToLetter(item), arr))
        return ''.join(arr)


# gets the task number using the task number region and the getNumber function
def getTaskNumber(task):
    _, cols = task.shape
    column1 = task[:, :int(cols/2.2)].copy()
    column2 = task[:, int(cols/2.2):].copy()
    tens = getNumber(column1, max=10, display=False)
    unit = getNumber(column2, max=10, display=False)
    return f'{outNumber(tens)}{outNumber(unit)}'


# gets the student number using the student number region and the getNumber function
def getStudentNumber(stNumber):
    rows, cols = stNumber.shape
    year1 = stNumber[
        : int(rows / 2.62),
        int(cols / 45): int(cols * (1/7) * 0.8)
    ]
    year2 = stNumber[
        : int(rows / 2.62),
        int(cols * (1/7) * 1.1): int(cols * (2/7) * 0.9)
    ]

    letter = stNumber[
        : int(rows / 1),
        int(cols * (2/7) * 1.1): int(cols * (3/7) * 0.9)
    ]
    digit1 = stNumber[
        : int(rows / 2.62),
        int(cols * (3/7) * 1.1): int(cols * (4/7) * 0.95)
    ]
    digit2 = stNumber[
        : int(rows / 2.62),
        int(cols * (4/7) * 1.05): int(cols * (5/7) * 0.95)
    ]
    digit3 = stNumber[
        : int(rows / 2.62),
        int(cols * (5/7) * 1.05): int(cols * (6/7) * 0.95)
    ]
    digit4 = stNumber[
        : int(rows / 2.62),
        int(cols * (6/7) * 1.05): int(cols * (7/7) * 0.95)
    ]

    # cv2.imshow('year1', cv2.resize(year1, (100, 500)))
    # cv2.imshow('year2', cv2.resize(year2, (100, 500)))
    # cv2.imshow('letter', cv2.resize(letter, (100, 1000)))
    # cv2.imshow('digit1', cv2.resize(digit1, (100, 500)))
    # cv2.imshow('digit2', cv2.resize(digit2, (100, 500)))
    # cv2.imshow('digit3', cv2.resize(digit3, (100, 500)))
    # cv2.imshow('digit4', cv2.resize(digit4, (100, 500)))

    y1 = getNumber(year1)
    y2 = getNumber(year2)
    l = getNumber(letter, max=26)
    d1 = getNumber(digit1)
    d2 = getNumber(digit2)
    d3 = getNumber(digit3)
    d4 = getNumber(digit4)
    return f'{outNumber(y1)}{outNumber(y2)}{numberToLetter(l)}{outNumber(d1)}{outNumber(d2)}{outNumber(d3)}{outNumber(d4)}'


# gets answers using the mcq region and the getNumber function
def getAnswers(groups):
    ret = []
    for i, group in enumerate(groups):
        rows, cols = group.shape
        group = group[int(rows / 6): int(rows / 1.1), :]
        rows, cols = group.shape
        questions = [group[int(20 + (rows * ((i - 1) / 5)))                           :int(rows * ((i) / 5)), :] for i in range(1, 6)]
        cv2.imshow(f'group', group)
        for j, question in enumerate(questions):
            rows, cols = question.shape
            cv2.imshow('question', question)
            answers = outList(
                getNumber(question, horizontal=True, max=5, multiple=True))
            ret.append(answers)
            # cv2.waitKey(0)
    return ret


# gets mcq groups of 5 questions for sheets with two columns
def groupsTwoColumn(mcq):
    rows, cols = mcq.shape
    column1 = mcq[:, int(cols / 6):int(cols / 2.5)]
    column2 = mcq[:, int(cols / 1.58):int(cols / 1.17)]

    # cv2.imshow('column1', cv2.resize(column1, (400, 1000)))
    # cv2.imshow('column2', cv2.resize(column2, (400, 1000)))

    rows, cols = column1.shape
    groups1 = [column1[int(rows * ((i - 1) / 6)):int(rows * ((i) / 6)), :] for i in range(1, 7)]
    groups2 = [column2[int(rows * ((i - 1) / 6)):int(rows * ((i) / 6)), :] for i in range(1, 7)]
    groups1.extend(groups2)
    return groups1


# gets mcq groups of 5 questions for sheets with three columns
def groupsThreeColumn(mcq):
    rows, cols = mcq.shape
    column1 = mcq[:, int(cols / 13):int(cols / 3.5)]
    column2 = mcq[:, int(cols / 2.3):int(cols / 1.55)]
    column3 = mcq[:, int(cols / 1.27):int(cols / 1)]
    cv2.imshow('column1', cv2.resize(column1, (400, 1000)))
    cv2.imshow('column2', cv2.resize(column2, (400, 1000)))
    cv2.imshow('column3', cv2.resize(column3, (400, 1000)))

    rows, cols = column1.shape
    groups1 = [column1[int(rows * ((i - 1) / 6)):int(rows * ((i) / 6)), :] for i in range(1, 7)]
    groups2 = [column2[int(rows * ((i - 1) / 6)):int(rows * ((i) / 6)), :] for i in range(1, 7)]
    groups3 = [column3[int(rows * ((i - 1) / 6)):int(rows * ((i) / 6)), :] for i in range(1, 7)]
    groups1.extend(groups2)
    groups1.extend(groups3)
    return groups1


# checks if a sheet has 3 columns
def threeColumnCheck(img):
    rows, cols = img.shape
    check = img[int(rows/2):int(rows/1.2), int(cols/3.15):int(cols/3)]
    check = cv2.resize(check, (500, 1000))
    _, check = cv2.threshold(
        check, 127, 255, cv2.THRESH_BINARY)
    check = cv2.bitwise_not(check)
    rows, cols = check.shape
    cv2.imshow('check', check)
    nPixels = rows * cols
    if cv2.countNonZero(check) > (nPixels * 0.1):
        return True
    return False


# write the student number, task number and answers for each sheet into a csv file
# each sheet is a single line
def writeToFile(studentNumber, taskNumber, answers, folder, filename):
    if not os.path.isdir(f'CSV/{folder}'):
        os.mkdir(f'CSV/{folder}')
    f = open(f'CSV/{folder}/{filename}.csv', 'w')
    f.write(f'{studentNumber},{taskNumber},')
    f.write(','.join(answers))


# does what it says
def doAllTheThings(dataset):
    for f in sorted(os.listdir(f'Sheets/{dataset}'), key=lambda i: i[6:10]):
        sheet = cv2.imread(f'Sheets/{dataset}/{f}', 1)
        cropped = cropAndCorrect(sheet)
        mcqs = multipleChoiceRegion(cropped)
        studentNumber = studentNumberRegion(cropped)
        task = taskFromStudentNumber(studentNumber)
        studentNumber = removeTaskFromStudentNumber(studentNumber)
        taskNumber = getTaskNumber(task)
        sNum = getStudentNumber(studentNumber)
        answers = None
        if threeColumnCheck(cropped):
            answers = getAnswers(groupsThreeColumn(mcqs))
        else:
            answers = getAnswers(groupsTwoColumn(mcqs))

        filename = f[:-4]
        writeToFile(sNum, taskNumber, answers, dataset, filename)
        cropped = cv2.resize(cropped, (800, 1000))
        mcqs = cv2.resize(mcqs, (800, 1000))
        studentNumber = cv2.resize(studentNumber, (400, 500))
        task = cv2.resize(task, (400, 500))
        print(sNum)
        print(taskNumber)
        print(','.join(answers))
        print(filename, end='\n\n')
        cv2.imshow('cropped', cropped)
        cv2.imshow('studentNumber', studentNumber)
        cv2.imshow('task', task)
        cv2.imshow('mcq', mcqs)
        cv2.waitKey(0)
        #k = cv2.waitKey(0)
        # if k == ord('q'):
        #    break
        cv2.destroyAllWindows()
    cv2.destroyAllWindows()


doAllTheThings('2018')
doAllTheThings('600dpi')
