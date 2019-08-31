import numpy as np
import cv2
import os
from astropy.io import fits
import pickle
import image_arithematic as ia
import matplotlib.pyplot as plt


#============================================================================
#                       LOAD THE MEAN OF DARK AND FLAT IMAGES
#============================================================================
file = open('mean_dark','rb')
mean_dark = pickle.load(file)
file.close()
file2 = open('mean_flat','rb')
mean_flat = pickle.load(file2)
file2.close()
#=============================================================================

base_path_data = "E:\MASTEMP"
destination_path = "E:\MASTFILTERED"

#print(mean_flat)
#print("============================================================================")
#print(mean_dark)
#print("==============================================================================")
#print(ia.diff_mat(mean_flat,mean_dark))
#print("===============================================================================")
#print(ia.image_mean(mean_flat))



i=0
#for file in os.listdir(base_path_data):
#    i+=1
#    hdulist = fits.open(os.path.join(base_path_data, file))
#    image = hdulist[0].data
#    num = (np.divide(ia.diff_mat(image , mean_dark) , ia.diff_mat(mean_flat , mean_dark)) * ia.image_mean(mean_flat))
#    den = ia.diff_mat(mean_flat , mean_dark)
#    new_image = np.divide(num,den)
#    new_name = str(i) + '.fits'
#    hdu = fits.PrimaryHDU(new_image)
#    hdul = fits.HDUList([hdu])
#    hdul.writeto(os.path.join(destination_path,new_name))


volume = np.zeros((2048,2060,12))
volume2= np.zeros((2048,2060))
j=0

for file in os.listdir(destination_path):
    j+=1
    hdulist = fits.open(os.path.join(destination_path, file))
    image = hdulist[0].data
    volume[:,:,j] = image
    cv2.imshow('dara', volume[:, :, j])
    cv2.waitKey(250)
   # cv2.destroyAllWindows()

cv2.destroyAllWindows()
    #plt.figure()
    #plt.imshow(image, cmap='gray')
    #plt.colorbar()
    #plt.close()




print("done writing the files ")

print(volume.shape)



print("done")
