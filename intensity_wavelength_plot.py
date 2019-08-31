from astropy.io import fits
import pickle
import numpy as np
import os
import cv2
import image_arithematic as ia
import matplotlib.pyplot as plt
#====================================================================================
#    input the data images
#    divide in batches of 20 images
#    calculate the intensity array and wavelenth array
#====================================================================================
#                                   DEFAULT PARAMETERS
#====================================================================================
WAVEID  =              854.209  # center wavelength
FP1VTR  =             4.6E-005  # FP1 Tuning rate
#=====================================================================================
#                                   NEEDED PARAMETERS
#=====================================================================================
#       FP1VOL  / volatage required for tuning
#       req. wavelength = WAVEID - (FP1VOL * FP1VTR)
#======================================================================================
WAVAELENGTH_ARRAY = []
INTENSITY_ARRAY = []
X_MAX = 200
Y_MAX = 200
SIZE_X = 2048
SIZE_Y = 2060
BATCH_SIZE = 20
BATCH_BUFFER = np.zeros((SIZE_X,SIZE_Y,BATCH_SIZE))
BASE_DIRECTORY = "E:\DATAMASTCA10052019\DATA"#"E:\MASTCAFE31052019_042749"
count=0

for file in os.listdir(BASE_DIRECTORY):
    hdul = fits.open(os.path.join(BASE_DIRECTORY , file))
    count += 1
    BATCH_BUFFER[:,:,(count-1)] = hdul[0].data
    if (count == 20):
        wavelength = ia.wavelength_of_fits(hdul)
        WAVAELENGTH_ARRAY.append(wavelength)
        intensity = ia.intensty_of_matrix(ia.mean_of_volume(BATCH_BUFFER), X_MAX ,Y_MAX)
        INTENSITY_ARRAY.append(intensity)
        count=0
        BATCH_BUFFER[:,:,:] = 0                                                                             # just for safety

file = open('wavelength_array','wb')
pickle.dump(WAVAELENGTH_ARRAY, file)
file.close()
file2 = open('intensity_array','wb')
pickle.dump(INTENSITY_ARRAY, file2)
file2.close()

print("done calculating the wavelength and intensity !!")
print(len(WAVAELENGTH_ARRAY))
print(len(INTENSITY_ARRAY))
































