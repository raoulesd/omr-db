import os
import shutil
from os import listdir
from os.path import isfile, join
from genericpath import isfile
import dearpygui.dearpygui as dpg
import cv2 as cv
import cv2 as cv2
import cv2.aruco as aruco
import numpy as np
import argparse

DEBUG = 0
TESSERACT = 0
DB9 = 0
DBIYO = 0

# Add some argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")
parser.add_argument("-d", "--debug", help="show debugging output",
                    action="store_true")
parser.add_argument("-o", "--ocr", help="enable tesseract",
                    action="store_true")
parser.add_argument("-p", "--profile", type=str, choices=['db9','dbiyo'],
                    help="profile to use for reading scores")
args = parser.parse_args()
if args.verbose:
    print("verbosity turned on")

if args.debug:
    print("debug turned on")
    DEBUG = 1

if args.ocr:
    print("tesseract enabled")
    TESSERACT = 1

if args.profile:
    if args.profile == 'db9':
        DB9 = 1
    elif args.profile == 'dbiyo':
        DBIYO = 1
    else:
        DB9 = 1


if TESSERACT == 1:
    import pytesseract
    from pytesseract import image_to_string

if DB9 == 1:
    COLUMNS = 9
    ROWS = 20
    ANSWERS = 3
    PRINTZ = 10.8
    PRINTT = 11.0
    PERC = 32.0

#PRINTZ = 9.7
#PRINTT = 10.5

if __name__ == '__main__':
    paths = ["processed", "toscan", "errored"]
    for p in paths:
        isExist = os.path.exists(p)
        if not isExist:
            os.makedirs(p)
            print(f"Made dir: {p}")

    path = "./toscan"
    fileList = [join(path, f) for f in os.listdir(path) if isfile(join(path, f))]
    print(fileList)

    csvFile = open("results.csv", "a")

    dpg.create_context()
    dpg.create_viewport(title='Review scores', width=1400, height=1400)
    dpg.setup_dearpygui()


    def get_next_file(isInitialization):
        global filename, fn_nopath, number, img, boulders, amountZT, triesZT, frame, data, texture_data, participant_name ,participant_number, participant_gender
        if not isInitialization:
            shutil.move(filename, "./processed", copy_function=shutil.copy2)

        path = "./toscan"
        fileList = [join(path, f) for f in os.listdir(path) if isfile(join(path, f))]
        if len(fileList) == 0:
            exit(0)

        sorted_fileList = sorted(fileList)
        filename = sorted_fileList.pop(0)
        fn_nopath = filename.split('/')[-1]
        fn_noext = fn_nopath.split('.')[0] 
        number = fn_noext.split('_')[-1]
        print(filename)
        print(fn_nopath)
        print(number)
        img, boulders, participant_name, participant_number, participant_gender = read_file(filename)
        amountZT, triesZT = getAmountAndTries(boulders)

        scale_down = 0.6
        frame = cv.resize(img, None, fx=scale_down, fy=scale_down, interpolation=cv.INTER_LINEAR)

        data = np.flip(frame, 2)  # because the camera data comes in as BGR and we need RGB
        data = data.ravel()  # flatten camera data to a 1 d stricture
        data = np.asfarray(data, dtype='f')  # change data type to 32bit floats
        texture_data = np.true_divide(data, 255.0)  # normalize image data to prepare for GPU
        try:
            dpg.set_value("texture_tag", texture_data)
            for i in range(1, ROWS + 1):
                dpg.set_value(f"zone_{i}", str(boulders[i - 1][1]))
                dpg.set_value(f"tops_{i}", str(boulders[i - 1][2]))
                update_amount_and_tries()
            dpg.set_value("user_name", f"{participant_name}")
            dpg.set_value("user_number", f"{participant_number}")
            dpg.set_value("gender", f"{participant_gender}")
        except:
            print("Booting up")


    def read_file(filename):
# these are defined globally        
#        COLUMNS = 9
#        ROWS = 20
#        ANSWERS = 3

        epsilon = 10  # image error sensitivity

        # load tracking tags
        tags = [cv2.imread("./markers/top_left.png", cv2.IMREAD_GRAYSCALE),
                cv2.imread("./markers/top_right.png", cv2.IMREAD_GRAYSCALE),
                cv2.imread("./markers/bottom_left.png", cv2.IMREAD_GRAYSCALE),
                cv2.imread("./markers/bottom_right.png", cv2.IMREAD_GRAYSCALE)]

        scaling = [869.0, 840.0]  # scaling factor for 8.5in. x 11in. paper
        columns = [[49.4 / scaling[0], 62.0 / scaling[1]]]  # dimensions of the columns of bubbles
        colspace = 78.5 / scaling[0]
        radius = 7.0 / scaling[0]  # radius of the bubbles
        spacing = [25.50 / scaling[0], 20.15 / scaling[1]]  # spacing of the rows and columns

        # Load the image from file
        img = cv2.imread(filename)
        height, width, channels = img.shape

        def FindCorners(paper, drawRect):
            corners = [[], [], [], []]
            gray_paper = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)  # convert image of paper to grayscale

            # scaling factor used later
            ratio = height / width

            # error detection
            if ratio == 0:
                return -1

            dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
            parameters = cv.aruco.DetectorParameters()
            detector = cv.aruco.ArucoDetector(dictionary, parameters)

            markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(gray_paper)
            if DEBUG == 1:
                print ("corners")
                print (markerCorners)
                print ("Ids")
                print (markerIds)
                print ("rejected")
                print (rejectedCandidates)
                cv2.namedWindow("debug_window", cv.WINDOW_GUI_NORMAL)
                cv2.resizeWindow("debug_window", 300, 700)
                aruco.drawDetectedMarkers(paper, rejectedCandidates)
                aruco.drawDetectedMarkers(paper, markerCorners)
                cv2.imshow('debug_window', paper)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            for corner, id in zip(markerCorners, markerIds):
                # print(corner[:,0][0])
                if id == 3:
                    corners[1] = [int(corner[0, :,0].mean()), int(corner[0, :, 1].mean())]
                elif id == 2:
                    corners[3] = [int(corner[0, :, 0].mean()), int(corner[0, :, 1].mean())]
                elif id == 1:
                    corners[2] = [int(corner[0, :, 0].mean()), int(corner[0, :, 1].mean())]
                elif id == 4:
                    corners[0] = [int(corner[0, :, 0].mean()), int(corner[0, :, 1].mean())]

            print ("Corners:")
            print(corners)
            if len(corners[0]) == 0:
                corners[0] = [corners[2][0],corners[1][1]]
            elif len(corners[1]) == 0:
                corners[1] = [corners[3][0],corners[0][1]]
            elif len(corners[2]) == 0:
                corners[2] = [corners[0][0],corners[3][1]]
            elif len(corners[3]) == 0:
                corners[3] = [corners[1][0],corners[2][1]]

            # draw the rectangle around the detected markers
            if drawRect:
                for corner in corners:
                    cv2.circle(paper, (corner[0], corner[1]), 10, (0, 0, 255), -1)

            return corners

        corners = FindCorners(img, False)
        print(corners)

        desired_points = np.float32([[68, 200], [1489, 200], [68, 1800], [1489, 1800]])
        points = np.float32(corners)

        M = cv2.getPerspectiveTransform(desired_points, desired_points)
        sheet = cv2.warpPerspective(img, M, (1589, 1900))

        #img = sheet
        height, width, channels = img.shape
        corners = FindCorners(img, True)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Threshold the image to binarize it
        treshImg, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        xdis, ydis = corners[3][0] - corners[1][0], corners[3][1] - corners[1][1]
        answersBoundingbox = [(int(corners[0][0] + 0.035 * xdis), corners[0][1] + int(0.055 * ydis)),
                              (corners[3][0] - int(0.15 * xdis), corners[3][1] - int(0.21 * ydis))]
        # cv2.rectangle(img, answersBoundingbox[0],
        #               answersBoundingbox[1], (0, 255, 0), thickness=2, lineType=8, shift=0)

        cv2.rectangle(img, (150,40), (850, 130), (0,255,0), thickness=2, lineType=8)
        if TESSERACT == 1:
            print(imageToText(img[60:130, 910:1030]))
            participant_number = imageToText(img[60:130, 910:1030]).strip().strip("\n")
            participant_name = imageToText(img[40:130, 150:850]).strip().strip("\n")
            participant_gender = imageToText(img[40:130, 150:850]).strip().strip("\n")
        else:
            participant_number = number
            participant_name = "James"
            participant_gender = "M"

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

                    percentile = (np.sum(roi == 255) / ((y2 - y1) * (x2 - x1))) * 100

                    if percentile > PERC:
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
            x1 = int((columns[0][0] + colspace * PRINTZ) * dimensions[0] + corners[0][0])
            y1 = int((columns[0][1] + 0.005 + i * spacing[1]) * dimensions[1] + corners[0][1])
            cv2.putText(img, str(boulders[i][1]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 190, 0), 2)

            x2 = int((columns[0][0] + colspace * PRINTT) * dimensions[0] + corners[0][0])
            cv2.putText(img, str(boulders[i][2]), (x2, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)

        #return img, boulders
        return img, boulders, participant_name, participant_number, participant_gender


    def getAmountAndTries(boulders):
        amountZT = [0, 0]
        triesZT = [0, 0]
        for i in range(0, len(boulders)):
            triesZT[0] += boulders[i][1]
            triesZT[1] += boulders[i][2]
            if boulders[i][1] != 0:
                amountZT[0] += 1
            if boulders[i][2] != 0:
                amountZT[1] += 1
        return amountZT, triesZT


    def export_to_csv(sender, callback):
        global boulders, amountZT, triesZT, filename, fn_nopath, participant_name, participant_number, participant_gender
        p_name = dpg.get_value("user_name")
        p_number = dpg.get_value("user_number")
        p_gender = dpg.get_value("gender")
        exportString = f"{p_name},{p_number},{p_gender},"
#        exportString += filename[9:]
        exportString += fn_nopath
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
        csvFile.write(f"{exportString}\n")
        csvFile.flush()
        get_next_file(False)

    if TESSERACT:
        def imageToText(img):
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thr = cv2.threshold(gry, 100, 255,
                                cv2.THRESH_BINARY_INV)[1]
            txt = image_to_string(thr, lang='eng', config='--psm 6')
            return txt

    def update_amount_and_tries():
        global amountZT, triesZT
        dpg.set_value("zone_total", amountZT[0])
        dpg.set_value("tops_total", amountZT[1])
        dpg.set_value("zone_tries", triesZT[0])
        dpg.set_value("tops_tries", triesZT[1])


    def set_boulders(sender, app_data, user_data):
        global boulders, amountZT, triesZT
        try:
            boulders[user_data[0]][user_data[1]] = int(dpg.get_value(item=sender))
            amountZT, triesZT = getAmountAndTries(boulders)
            update_amount_and_tries()
        except ValueError:
            print("no change since no integer was inputted")


    get_next_file(True)

    with dpg.texture_registry(show=False):
        dpg.add_raw_texture(frame.shape[1], frame.shape[0], texture_data, tag="texture_tag",
                            format=dpg.mvFormat_Float_rgb)

    with dpg.window(label="resultstester", tag="mainWindow"):
        with dpg.table(header_row=False):
            dpg.add_table_column(width_stretch=True)
            dpg.add_table_column(width_fixed=True, init_width_or_weight=250.0)
            with dpg.table_row():
                with dpg.table_cell():
                    dpg.add_image("texture_tag")
                with dpg.table_cell():
                    dpg.add_text(f"Naam kandidaat:")
                    dpg.add_input_text(tag=f"user_name", default_value=participant_name)
                    dpg.add_text(f"Nummer kandidaat:")
                    dpg.add_input_text(tag=f"user_number", default_value=participant_number)
                    dpg.add_text(f"Geslacht kandidaat:")
                    dpg.add_input_text(tag=f"gender", default_value=participant_gender)
                    with dpg.table(header_row=False):
                        dpg.add_table_column(width_fixed=True)
                        dpg.add_table_column(width_fixed=True)
                        dpg.add_table_column(width_fixed=True, init_width_or_weight=40)
                        dpg.add_table_column(width_fixed=True)
                        dpg.add_table_column(width_fixed=True, init_width_or_weight=40)
                        for i in range(1, ROWS + 1):
                            with dpg.table_row():
                                dpg.add_text(f"B{i}")
                                dpg.add_text("Z")
                                dpg.add_input_text(tag=f"zone_{i}", default_value=str(boulders[i - 1][1]),
                                                   callback=set_boulders, user_data=[i - 1, 1], decimal=True)
                                dpg.add_text("T")
                                dpg.add_input_text(tag=f"tops_{i}", default_value=str(boulders[i - 1][2]),
                                                   callback=set_boulders, user_data=[i - 1, 2], decimal=True)
                        with dpg.table_row():
                            dpg.add_text("Aantal")
                            dpg.add_text("Z")
                            dpg.add_input_text(tag=f"zone_total", default_value=str(amountZT[0]))
                            dpg.add_text("T")
                            dpg.add_input_text(tag=f"tops_total", default_value=str(amountZT[1]))
                        with dpg.table_row():
                            dpg.add_text("Pogingen")
                            dpg.add_text("Z")
                            dpg.add_input_text(tag=f"zone_tries", default_value=str(triesZT[0]))
                            dpg.add_text("T")
                            dpg.add_input_text(tag=f"tops_tries", default_value=str(triesZT[1]))
                    dpg.add_button(label="export", callback=export_to_csv)

    dpg.show_viewport()
    dpg.set_primary_window("mainWindow", True)
    dpg.start_dearpygui()
    dpg.destroy_context()
