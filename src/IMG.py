import cv2
import glob
import numpy as np


def X_data():
    X_data = []
    files = glob.glob ("E:/SimpleHTR-master/data/words/a01/a01-000u/*.PNG")
    for myFile in files:
        image = cv2.imread (myFile)
        X_data.append (image)
        return(X_data)

# -*- coding: utf-8 -*-

