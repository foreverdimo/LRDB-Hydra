import cv2 as cv
import numpy as np
import math


def HPF(img):
    rows, cols, ___ = img.shape()
    img_low = cv.resize(img, None,fx = 1/16, fy = 1/16)
    img_low = cv.resize(img_low, None,fx = 16, fy = 16, interpolation= cv.INTER_CUBIC)
    img = (img - img_low)/255
    return img


def Error_Map(reference, distorted, epsilon):
    return (1/np.log2(255**2/epsilon))*np.log2( 1/((reference - distorted)**2 + epsilon/(255**2)) )
