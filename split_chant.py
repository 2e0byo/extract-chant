#!/usr/bin/python3
# based on code from https://stackoverflow.com/questions/34981144/split-text-lines-in-scanned-document
import argparse

import cv2

import hougher
import line_splitter as ls

color_index = 0
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]


def colorise():
    """toggle colour"""
    global color_index
    color_index += 1
    return colors[color_index % len(colors)]


parser = argparse.ArgumentParser()
parser.add_argument("INPUT", help="Input Filename")
parser.add_argument(
    "-p",
    "--prefix",
    help="Prefix for saved images, default is 'line'",
    default="line")
parser.add_argument(
    "--offset",
    help="Offset for numbering",
    type=int,
    default=0
)
parser.add_argument(
    "-r",
    "--rectangle",
    help="Use minimal area rectangle to rotate not houghing",
    action="store_true")
parser.add_argument(
    "--min_line_length",
    help="Min Line Length for Houghing",
    type=int,
    default=hougher.min_line_length)
parser.add_argument(
    "--max_line_gap",
    help="Max Line Gap for Houghing",
    type=int,
    default=hougher.max_line_gap)
parser.add_argument(
    "--threshold", help="Threshold for Houghing", type=int, default=hougher.th)
parser.add_argument(
    "--hough_detect_lines",
    help="Use Houghing to check each line, only exporting if four or more lines detected (=stave)",
    action="store_true")

parser.add_argument(
    "--min-white",
    help="Minimum white length",
    type=int,
    default=ls.min_white)

args = parser.parse_args()

# Execution starts here

ls.min_white = args.min_white

img, gray, threshed = ls.read_image(args.INPUT)
if args.rectangle is True:
    ang, M = ls.min_area_rect_rotation(threshed)
else:
    ang, M = hougher.hough_image(
        args.threshold,
        args.min_line_length,
        args.max_line_gap,
        hougher.process_image(hougher.load_image(args.INPUT))[1],
        img,
        draw=False)
rotated, rotated_original = ls.rotate_image_and_threshed(M, threshed, img)
uppers_y, lowers_y, lower_x, upper_x = ls.get_lines(rotated)
bands = ls.smarten_lines(uppers_y, lowers_y)

i = args.offset
for band in bands:
    print(band)
    if args.hough_detect_lines is True:
        tmp = rotated_original[band[0]:band[1], lower_x:upper_x].copy()
        if not hougher.hough_detect_lines(tmp):
            continue
    i += 1
    cv2.imwrite("%s%02i.png" % (args.prefix, i),
                rotated_original[band[0]:band[1], lower_x:upper_x])

# for (up, low) in bands:
#     cv2.rectangle(rotated_original, (lower_x, up), (upper_x, low),
#                   colorise(), 2)

# cv2.imwrite("result.png", rotated_original)
