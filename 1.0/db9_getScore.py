import math

import cv2 as cv2
import numpy as np

def getScore(filename):
    COLUMNS = 9
    ROWS = 20
    ANSWERS = 3

    epsilon = 10 #image error sensitivity

    # load tracking tags
    tags = [cv2.imread("markers/top_left.png", cv2.IMREAD_GRAYSCALE),
            cv2.imread("markers/top_right.png", cv2.IMREAD_GRAYSCALE),
            cv2.imread("markers/bottom_left.png", cv2.IMREAD_GRAYSCALE),
            cv2.imread("markers/bottom_right.png", cv2.IMREAD_GRAYSCALE)]

    scaling = [869.0, 840.0] #scaling factor for 8.5in. x 11in. paper
    columns = [[52.8 / scaling[0], 63.2 / scaling[1]]] #dimensions of the columns of bubbles
    colspace = 77.5 /scaling[0]
    radius = 6.0 / scaling[0] #radius of the bubbles
    spacing = [25.02 / scaling[0], 20.20 / scaling[1]] #spacing of the rows and columns

    # Load the image from file
    img = cv2.imread(filename)
    height, width, channels = img.shape


    corners = []  # array to hold found corners


    def FindCorners(paper, drawRect):
        gray_paper = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY) #convert image of paper to grayscale

        #scaling factor used later
        ratio = height / width

        #error detection
        if ratio == 0:
            return -1

        #try to find the tags via convolving the image
        for tag in tags:
            tag = cv2.resize(tag, (0,0), fx=ratio, fy=ratio) #resize tags to the ratio of the image

            #convolve the image
            convimg = (cv2.filter2D(np.float32(cv2.bitwise_not(gray_paper)), -1, np.float32(cv2.bitwise_not(tag))))

            #find the maximum of the convolution
            corner = np.unravel_index(convimg.argmax(), convimg.shape)

            #append the coordinates of the corner
            corners.append([corner[1], corner[0]]) #reversed because array order is different than image coordinate

        #draw the rectangle around the detected markers
        if drawRect:
            for corner in corners:
                cv2.rectangle(paper, (corner[0] - int(ratio * 25), corner[1] - int(ratio * 25)),
                (corner[0] + int(ratio * 25), corner[1] + int(ratio * 25)), (0, 255, 0), thickness=2, lineType=8, shift=0)

        #check if detected markers form roughly parallel lines when connected
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

    desired_points = np.float32([[68, 424], [1489, 424], [68, 1797], [1489, 1797]])
    points = np.float32(corners)

    M = cv2.getPerspectiveTransform(points, desired_points)
    sheet = cv2.warpPerspective(img, M, (1589, 1997))


    img = sheet
    height, width, channels = img.shape
    corners = []
    FindCorners(img, True)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the image to binarize it
    treshImg, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)

    xdis, ydis = corners[3][0] - corners[0][0], corners[3][1] - corners[0][1]
    answersBoundingbox = [(int(corners[0][0] + 0.035 *xdis), corners[0][1] + int(0.055 *ydis)), (corners[3][0] - int(0.15 * xdis), corners[3][1] - int(0.21 * ydis))]
    cv2.rectangle(img, answersBoundingbox[0],
            answersBoundingbox[1], (0, 255, 0), thickness=2, lineType=8, shift=0)

    # calculate dimensions for scaling
    dimensions = [corners[1][0] - corners[0][0], corners[2][1] - corners[0][1]]


    boulders = list()
    for i in range(0, ROWS):
        boulders.append([0,0,0])

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

                percentile = (np.sum(roi == 255)/((y2-y1) * (x2-x1))) * 100

                # if percentile > 40.0:
                #     if (j != 0 and boulders[i][j] == 0) or j == 0:
                #         boulders[i][j] = k + 1
                if percentile > 40.0:
                    print(percentile)
                    if j == 0:
                        boulders[i][j] = k + 1
                    if j == 1 and boulders[i][1] == 0:
                        boulders[i][j] = k + 1
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=2, lineType=8, shift=0)
                    if j == 2:
                        boulders[i][j] = k + 1
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=2, lineType=8, shift=0)
                        if boulders[i][1] == 0:
                            boulders[i][1] = k + 1

    for i in range(0, ROWS):
        x1 = int((columns[0][0] + colspace * 9.7) * dimensions[0] + corners[0][0])
        y1 = int((columns[0][1]+0.005 + i * spacing[1]) * dimensions[1] + corners[0][1])
        cv2.putText(img, str(boulders[i][1]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 190, 0), 2)

        x2 = int((columns[0][0] + colspace * 10.5) * dimensions[0] + corners[0][0])
        cv2.putText(img, str(boulders[i][2]), (x2, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 190, 0), 2)

    # Show the image with the rectangles
    finish = False
    change = False
    changeIndex = -1
    currind = 0
    name = ""
    # cv2.setMouseCallback('results', draw_circle)
    while not finish:
        cv2.imshow("results", cv2.resize(img, (0, 0), fx=0.7, fy=0.7))
        key = cv2.waitKey(0)
        if key == ord('w'):  # arrow key up
            if currind < ROWS-1:
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
            x1 = int((columns[0][0] + colspace * (8.2 + 1.7 * changeIndex) * dimensions[0] + corners[0][0]))
            y1 = int((columns[0][1] + 0.005 + currind * spacing[1]) * dimensions[1] + corners[0][1])
            cv2.putText(img, str(boulders[currind][changeIndex]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255), 2)
            change = False
            changeIndex = -1
        elif key == ord('n'):
            isNameDone = False
            while not isNameDone:
                cv2.rectangle(img, (0, 0), (width, 150), (255, 255, 255), -1)
                cv2.putText(img, f"input name: {name}", (30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                cv2.imshow("results", cv2.resize(img, (0, 0), fx=0.7, fy=0.7))
                local_key = cv2.waitKey(0)
                print(local_key)
                if local_key == 13: # enter
                    isNameDone = True
                elif local_key == 8: #backspace
                    name = name[:-1]
                else:
                    name = f"{name}{chr(local_key)}"
        elif key == ord('d'):
            print("done!")
            finish = True
        cv2.rectangle(img, (0,0), (width,150), (255, 255, 255), -1)
        cv2.putText(img, f"input name: {name}", (30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(img, f"B{str(currind+1)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        if change:
            cv2.putText(img, "change", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            if changeIndex == 1:
                cv2.putText(img, "zone", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            elif changeIndex == 2:
                cv2.putText(img, "top", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    exportString = f"{name},"
    exportString += filename[9:]
    amountZT = [0, 0]
    triesZT = [0, 0]
    for i in range(0, len(boulders)):
        triesZT[0] += boulders[i][1]
        triesZT[1] += boulders[i][2]
        if boulders[i][1] != 0:
            amountZT[0] += 1
        if boulders[i][2] != 0:
            amountZT[1] += 1
        exportString += f",B{i + 1} T{boulders[i][2]}Z{boulders[i][1]}"
    exportString += f",{amountZT[1]},{amountZT[0]}"
    exportString += f",{triesZT[1]},{triesZT[0]}"
    print(exportString)
    cv2.destroyAllWindows()
    return exportString


    # Show the image with the rectangles
    cv2.imwrite(f"checkboxes_{filename}", img)
    cv2.imshow("Checkboxes", cv2.resize(img, (0, 0), fx=0.7, fy=0.7))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
