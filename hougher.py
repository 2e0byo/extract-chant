#!/usr/bin/python3
from sys import argv
import cv2
import numpy as np

#edges = cv2.Canny(gray, 50, 150, apertureSize=7)
min_line_length = 246
max_line_gap = 7
th = 22


def nothing(x):
    return


def hough_detect_lines(img):
    """hough a fragment to detect four lines if its a stave"""
    img_orig, threshed = process_image(img)
    lines = cv2.HoughLinesP(threshed, 1, np.pi / 180, th, 0, min_line_length,
                            max_line_gap)
    if lines is None:
        return(False)
    if len(lines) >= 4:
        return (True)
    else:
        return (False)


def hough_image(th, min_line_length, max_line_gap, threshed, img, draw=True):
    lines = cv2.HoughLinesP(threshed, 1, np.pi / 180, th, 0, min_line_length,
                            max_line_gap)
    angles = []
    if lines is None:
        print('no lines found')
        return (False, False)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if draw is True:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        oh = (y2 - y1) / (x2 - x1)
        angles.append(np.arctan(oh) * 180 / np.pi)
    angles = np.array(angles)
    angles = np.array(
        [i for i in angles])# if abs(angles.mean() - i) < (angles.std() * 1.5)])
    ang = angles.mean()
    print("angle hough:", ang)
    cx = round(img.shape[1] / 2)
    cy = round(img.shape[0] / 2)
    M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    return (img, M)


def load_image(fname):
    """Load image for houghing at correct scale"""
    img_orig = cv2.imread(fname)
    return (img_orig)


def process_image(img_orig):
    # get a reasonable sixed version:
    reasonable_size = 1000  # max length
    # if len([i for i in img_orig.shape[:2] if i > reasonable_size]) > 0:
    scale = reasonable_size / max(img_orig.shape[:2])
    #    h,w = [i * scale for i in img_orig.shape[:2]]
    img_orig = cv2.resize(img_orig, (0, 0), fx=scale, fy=scale)

    gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
    t, threshed = cv2.threshold(gray, 127, 255,
                                cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return (img_orig, threshed)


if __name__ == '__main__':
    img_orig, threshed = process_image(load_image(argv[1]))
    cv2.namedWindow("houghing")
    cv2.createTrackbar("Min Line Length", "houghing", min_line_length, 1000,
                       nothing)
    cv2.createTrackbar("Max Line Gap", "houghing", max_line_gap, 150, nothing)
    cv2.createTrackbar("Threshold", "houghing", th, 400, nothing)

    while (1):
        img = img_orig.copy()
        img, M = hough_image(th, min_line_length, max_line_gap, threshed, img)
        if img is not False:
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
            cv2.imshow("Houghing Test", img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        min_line_length = cv2.getTrackbarPos("Min Line Length", "houghing")
        max_line_gap = cv2.getTrackbarPos("Max Line Gap", "houghing")
        th = cv2.getTrackbarPos("Threshold", "houghing")
