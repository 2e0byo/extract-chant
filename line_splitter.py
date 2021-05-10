# based on code from
# https://stackoverflow.com/questions/34981144/split-text-lines-in-scanned-document
import cv2
import numpy as np

white_bleed = (.5, .5)  # percentage of white to add to selection (above,below)
min_white = 20  # minimum length of white pixels


def read_image(fname):
    """Read image and return image and threshed version for analysis"""
    img = cv2.imread(fname)
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        gray = img.copy()
    th, threshed = cv2.threshold(gray, 127, 255,
                                 cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return (img, gray, threshed)


def min_area_rect_rotation(threshed):
    """
    Find Min area rectangle of all non-zero pixels and return angle to
    rotate image to align vertically
    """
    pts = cv2.findNonZero(threshed)
    (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
    if w > h:  # rotate to portrait
        w, h = h, w
        ang += 90
    print("angle rect:", ang)
    M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    return (ang, M)


def rotate_image_and_threshed(M, threshed, img):
    """
    Rotate image by ang
    """
    rotated = cv2.warpAffine(threshed, M, (img.shape[1], img.shape[0]))
    rotated_original = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return (rotated, rotated_original)


def get_lines(img, th=False):
    """
    Get upper and lower boundary of each line in image by stepping
    through averaged histogram.  Threshold defaults to minimum
    value in hist (probably 0).
    """
    hist = cv2.reduce(img, 1, cv2.REDUCE_AVG).reshape(-1)
    if not th:
        th = min(hist)
    H, W = img.shape[:2]
    uppers_y = [y for y in range(H - 1) if hist[y] <= th and hist[y + 1] > th]
    lowers_y = [y for y in range(H - 1) if hist[y] > th and hist[y + 1] <= th]
    hist = cv2.reduce(img, 0, cv2.REDUCE_AVG).reshape(-1)
    if not th:
        th = min(hist)
    th = 1
    black_width = 100
    lower_x = min([
        x for x in range(W - black_width)
        if hist[x] <= th
        and all([hist[x + i + 1] > th for i in range(black_width)])
    ])

    upper_x = max([
        x for x in range(W - black_width)
        if hist[x] > th
        and all([hist[x + i + 1] <= th for i in range(black_width)])
    ])
    if len(lowers_y) < len(uppers_y):  # if ends with cut-off line
        uppers_y.pop()

    return (uppers_y, lowers_y, lower_x, upper_x)


def smarten_lines(uppers_y, lowers_y):
    """
    Add appropriate whitespace around lines and combine any which are too small
    """
    bands = []
    last_band = -1
    gaps = []
    for i in range(len(uppers_y)):
        if i > 0:
            gap = uppers_y[i] - lowers_y[i - 1]
            gaps.append(gap)
        else:
            gap = False

        if gap is not False and gap < min_white:
            bands[last_band][1] = lowers_y[i]
        else:
            if gap is False:
                gap = 0
            else:
                bands[last_band][1] += round(gap * white_bleed[0])
            bands.append(
                [uppers_y[i] - round(gap * white_bleed[0]), lowers_y[i]])
            last_band += 1
    # get mean gap for first/last band
    # excluding outliers (1.5*std away from mean)
    gaps = np.array(gaps)
    mean_gap = np.array(
        [i for i in gaps if abs(gaps.mean() - i) < (gaps.std() * 1.5)]).mean()

    bands[-1][1] += int(round(mean_gap * white_bleed[1]))
    bands[0][0] -= int(round(mean_gap * white_bleed[0]))
    if bands[0][0] < 0:
        bands[0][0] = 0

    return (bands)
