import numpy as np
import cv2
import os
from astropy.io import fits
import pickle
import image_arithematic as ia
import matplotlib.pyplot as plt


base_path_data = "E:\DATAMASTCA10052019\CORRECTED_X_Y"
j=0
volume = np.zeros((1110,1100,4))
for file in os.listdir(base_path_data):
    j+=1
    hdulist = fits.open(os.path.join(base_path_data, file))
    image = hdulist[0].data
    print(image.shape)
    volume[:,:,j] = image
    cv2.imshow('dara', volume[:,:,j])
    cv2.waitKey()
    #cv2.destroyAllWindows()

cv2.destroyAllWindows()