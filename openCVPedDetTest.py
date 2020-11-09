# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

# Testing OpenCV Pedestrian HOG + SVM Detection :
# SOURCE CODE FROM : https://www.pyimagesearch.com/2015/11/09/pedestrian-detection-opencv/
# Seems to work both some on the clear and noisy dataset (GRAZ 01 and UG2+) but on images from a distance
# Does not do as well on closer images or images of people sitting down
''' 
# imutils  installed, you’ll need to upgrade to the latest version (v0.3.1) which includes the implementation of the non_max_suppression  function
# reduces false positives - takes multple overlapping bounding boxes and reduces to single box
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# --images : arg which is the path to the directory that contains
# the list of images we are going to perform pedestrian detection on.
ap.add_argument("-i", "--images", required=True, help="path to images directory")
args = vars(ap.parse_args())
'''
# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
clearpath = r'D:\Computer Vision Project\GRAZ DATASET\Graz_01\persons\persons'

noisypath = r'D:\Computer Vision Project\UG Dataset\RTTS\JPEGImages'
# loop over the image paths
for imagePath in paths.list_images(clearpath):
    # load the image and resize it to (1) reduce detection time
    # and (2) improve detection accuracy
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=min(400, image.shape[1]))
    orig = image.copy()
    # detect people in the image
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
                                            padding=(8, 8), scale=1.05)
    # draw the original bounding boxes
    for (x, y, w, h) in rects:
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
    # show some information on the number of bounding boxes
    filename = imagePath[imagePath.rfind("/") + 1:]
    print("[INFO] {}: {} original boxes, {} after suppression".format(
        filename, len(rects), len(pick)))
    # show the output images
    cv2.imshow("Before NMS", orig)
    cv2.imshow("After NMS", image)
    cv2.waitKey(0)
