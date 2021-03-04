import argparse
import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
import csv
onlyfiles = [f for f in listdir("dataset") if isfile(join("dataset/", f))]
from PIL import Image
print(onlyfiles)
filesData = []
for f in listdir("dataset"):
    x,y=f.split(",")
    y,tipe=y.split(".")
    print(x,y)
    filesData.append((int(x),int(y)))
print(filesData)
pictureData = []
for file in onlyfiles:
    print(file)
    lower_blue= np.array([100,50,50])
    upper_blue = np.array([138,255,255])

    image = cv2.imread("dataset/"+file)
    cv2.imshow("image", image)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    cv2.imshow("Mask", mask)
    cnts, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    ((x, y), radius) = cv2.minEnclosingCircle(c)
    M = cv2.moments(c)
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    print(center)
    cv2.circle(image, center, 5, (0, 0, 255), -1)
    pictureData.append(center)
print(filesData)
print(pictureData)
with open("Y.csv", "a") as my_csv:
    csvWriter = csv.writer(my_csv, delimiter=',')
    csvWriter.writerows(filesData)
with open("X.csv", "a") as my_csv:
    csvWriter = csv.writer(my_csv, delimiter=',')
    csvWriter.writerows(pictureData)