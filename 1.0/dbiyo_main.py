import math
import os
import shutil
from os.path import join

import cv2 as cv2
import numpy as np


def getScore(filename):
    COLUMNS = 9
    ROWS = 20
    ANSWERS = 3

    epsilon = 10  # image error sensitivity
    filename = "../../AppData/Roaming/JetBrains/PyCharm2023.1/scratches/img20230922_17102421.jpg"

    # load tracking tags
    tags = [cv2.imread("markers/top_left.png", cv2.IMREAD_GRAYSCALE),
            cv2.imread("markers/top_right.png", cv2.IMREAD_GRAYSCALE),
            cv2.imread("markers/bottom_left.png", cv2.IMREAD_GRAYSCALE),
            cv2.imread("markers/bottom_right.png", cv2.IMREAD_GRAYSCALE)]

    scaling = [869.0, 840.0]  # scaling factor for 8.5in. x 11in. paper 1660 2347
    columns = [[78 / scaling[0], 40.5 / scaling[1]]]  # dimensions of the columns of bubbles
    colspace = 137.2 / scaling[0]
    radius = 7.5 / scaling[0]  # radius of the bubbles
    spacing = [36.2 / scaling[0], 22.35 / scaling[1]]  # spacing of the rows and columns

    # Load the image from file
    img = cv2.imread(filename)
    height, width, channels = img.shape

    corners = []  # array to hold found corners

    def FindCorners(paper, drawRect):
        gray_paper = paper  # cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY) #convert image of paper to grayscale

        # scaling factor used later
        ratio = height / width

        # error detection
        if ratio == 0:
            return -1

        # try to find the tags via convolving the image
        for tag in tags:
            tag = cv2.resize(tag, (0, 0), fx=ratio, fy=ratio)  # resize tags to the ratio of the image

            # convolve the image
            convimg = (cv2.filter2D(np.float32(cv2.bitwise_not(gray_paper)), -1, np.float32(cv2.bitwise_not(tag))))
            cv2.imshow("results", cv2.resize(convimg, (0, 0), fx=0.35, fy=0.35))

            # find the maximum of the convolution
            corner = np.unravel_index(convimg.argmax(), convimg.shape)

            # append the coordinates of the corner
            corners.append([corner[1], corner[0]])  # reversed because array order is different than image coordinate

        # draw the rectangle around the detected markers
        if drawRect:
            for corner in corners:
                cv2.rectangle(paper, (corner[0] - int(ratio * 25), corner[1] - int(ratio * 25)),
                              (corner[0] + int(ratio * 25), corner[1] + int(ratio * 25)), (0, 255, 0), thickness=2,
                              lineType=8, shift=0)

        # check if detected markers form roughly parallel lines when connected
        if corners[0][0] - corners[2][0] > epsilon:
            return None

        if corners[1][0] - corners[3][0] > epsilon:
            return None

        if corners[0][1] - corners[1][1] > epsilon:
            return None

        if corners[2][1] - corners[3][1] > epsilon:
            return None

        return

    FindCorners(img, False)
    print(corners)
    #
    desired_points = np.float32([[112, 350], [2282, 350], [112, 3310], [2282, 3310]])
    points = np.float32(corners)

    # M = cv2.getPerspectiveTransform(points, desired_points)
    # sheet = cv2.warpPerspective(img, M, (2500, 3520))
    #
    # img = sheet
    height, width, channels = img.shape
    corners = []
    FindCorners(img, True)

    gray = img  # cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the image to binarize it
    treshImg, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # calculate dimensions for scaling
    dimensions = [corners[1][0] - corners[0][0], corners[2][1] - corners[0][1]]

    boulders = list()
    for i in range(0, ROWS):
        boulders.append([0, 0, 0])

    # iterate over test questions
    for i in range(0, ROWS):  # rows
        for k in range(0, COLUMNS):  # columns
            for j in range(0, ANSWERS):  # answers
                # coordinates of the answer bubble
                x1 = int((columns[0][0] + colspace * k + j * spacing[0] - radius) * dimensions[0] + corners[0][0])
                y1 = int((columns[0][1] + i * spacing[1] - radius) * dimensions[1] + corners[0][1])
                x2 = int((columns[0][0] + colspace * k + j * spacing[0] + radius) * dimensions[0] + corners[0][0])
                y2 = int((columns[0][1] + i * spacing[1] + radius) * dimensions[1] + corners[0][1])

                # draw rectangles around bubbles
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), thickness=1, lineType=8, shift=0)

                roi = thresh[y1:y2, x1:x2]
                percentile = (np.sum(roi == 255) / (abs(y2 - y1) * abs(x2 - x1))) * 100

                # print(percentile)

                rect = False
                if percentile > 105.0:
                    if j == 0:
                        boulders[i][j] = k + 1
                    if j == 1 and boulders[i][1] == 0:
                        boulders[i][j] = k + 1
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=2, lineType=8, shift=0)
                        rect = True
                    if j == 2:
                        boulders[i][j] = k + 1
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=2, lineType=8, shift=0)
                        rect = True
                        if boulders[i][1] == 0:
                            boulders[i][1] = k + 1
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=2, lineType=8, shift=0)
                            rect = True
                # if not rect:
                # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), thickness=0, lineType=8, shift=0)

    for i in range(0, ROWS):
        x1 = int((columns[0][0] + colspace * 4.7) * dimensions[0] + corners[0][0])
        y1 = int((columns[0][1] + 0.005 + i * spacing[1]) * dimensions[1] + corners[0][1])
        cv2.putText(img, str(boulders[i][1]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 190, 0), 5)

        x2 = int((columns[0][0] + colspace * 5.9) * dimensions[0] + corners[0][0])
        cv2.putText(img, str(boulders[i][2]), (x2, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 190, 0), 5)

    # Show the image with the rectangles
    finish = False
    change = False
    changeIndex = -1
    currind = 0
    while not finish:
        cv2.imshow("results", cv2.resize(img, (0, 0), fx=0.35, fy=0.35))
        key = cv2.waitKey(0)
        if key == ord('w'):  # arrow key up
            if currind < 29:
                currind += 1
        elif key == ord('s'):  # arrow key down
            if currind > 0:
                currind -= 1
        elif key == ord('c'):  # change
            change = not change
        elif key == ord('z'):  # change
            if change:
                changeIndex = 1
        elif key == ord('t'):  # change
            if change:
                changeIndex = 2
        elif ord('0') <= key <= ord('9') and change and changeIndex != -1:
            boulders[currind][changeIndex] = key - ord('0')
            x1 = int((columns[0][0] + colspace * (4.0 + 1.2*changeIndex) * dimensions[0] + corners[0][0]))
            y1 = int((columns[0][1] + currind * spacing[1]) * dimensions[1] + corners[0][1])
            cv2.putText(img, str(boulders[currind][changeIndex]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 5)
            change = False
            changeIndex = -1
        elif key == ord('d'):
            print("done!")
            finish = True
        print(currind)

    exportString = ""
    exportString += filename[len(filename)-7:len(filename)-4]
    amountZT = [0,0]
    triesZT = [0,0]
    for i in range(0, len(boulders)):
        triesZT[0] += boulders[i][1]
        triesZT[1] += boulders[i][2]
        if boulders[i][1] != 0:
            amountZT[0] += 1
        if boulders[i][2] != 0:
            amountZT[1] += 1
        exportString += f",B{i+1} T{boulders[i][2]}Z{boulders[i][1]}"
    exportString += f",{amountZT[1]},{amountZT[0]}"
    exportString += f",{triesZT[1]},{triesZT[0]}"
    print(exportString)
    cv2.destroyAllWindows()
    return exportString

