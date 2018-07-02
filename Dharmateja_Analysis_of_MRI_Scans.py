import sys
import numpy as np
from numpy import *
from scipy import stats
import pandas as pd
import datetime
import math
import os.path
import matplotlib.pyplot as plt
import time
import pydicom
import dicom
import os
import numpy
from matplotlib import pyplot, cm
import mritopng
from os import listdir
from PIL import Image as PImage
from IPython.display import Image
from PIL import Image
import os, sys

path = "C:/Users/DTP/Desktop/MS Analytics/Quarter-3/SubQuarter-2/Python For Data Science/Scans"

dir = os.listdir(path)
size = 7016, 4961

# Function to convert complete folder of Images in DICOM files to PNG
def convert_to_png(path1,path2):
    mritopng.convert_folder(path1,path2)

#Function to get images from directory and resize with desired size or resolution
def resize_image(path,size):
    for i in dir:
        if os.path.isfile(path+item):
            image = Image.open(path+i)          #opening any particular image in a file
            f, e = os.path.splitext(path+i)
            re_image = image.resize(size, Image.ANTIALIAS)  #resizing the image with assigned size.
            re_image.save(f + ' resized.png', 'PNG', quality=90)    #saving the resized image in png format by assigning name.

# Function  to convert image unite to HU
def scan_HU(path):
    ct_image = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]  #reading DICOM Files from Directory
    ct_image.sort(key = lambda x: int(x.ImagePositionPatient[2]))

    #Adjusting image positioning
    try:
        ct_image_thickness = np.abs(ct_image[0].ImagePositionPatient[2] - ct_image[1].ImagePositionPatient[2])
    except:
        ct_image_thickness = np.abs(ct_image[0].SliceLocation - ct_image[1].SliceLocation)

    for i in ct_image:
        i.ct_Thickness = ct_image_thickness

    image = np.stack([i.pixel_array for i in ct_image])

    # Convert to int16  as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU) using the mathematical formulae
    for n in range(len(ct_image)):

        intercept = ct_image[n].RescaleIntercept  #Calculating Intercept
        slope = ct_image[n].RescaleSlope          #Calculating Slope

        if slope != 1:
            image[n] = slope * image[n].astype(np.float64)
            image[n] = image[n].astype(np.int16)

        image[n] += np.int16(intercept)

    return np.array(image, dtype=np.int16)


if __name__ == "__main__":
    # Convert a whole folder of DICOM to PNG recursively
    convert_to_png('C:/Users/DTP/Desktop/DICOM/', 'C:/Users/DTP/Desktop/PNG/')

    # Resize Images with better resolution
    resize_image(path,size)

    #Convert into Hounsfield Units
    scan_HU('C:/Users/DTP/Desktop/DICOM/')
