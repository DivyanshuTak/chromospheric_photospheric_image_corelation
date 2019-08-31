import numpy as np
import cv2
import os
from astropy.io import fits
import pickle
import image_arithematic as ia
import matplotlib.pyplot as plt

#=======================================================================================================================
#                                   METHOD FOR CALCULTAING DOPPLER SHIFT IN FREQUENCY
#=======================================================================================================================
'''
                                first take each fits image from the disc 
                                compress it to increase the SNR 
                                calculate the mean value for the compressed image
                                loop for each pixel calculating the ratio of that 
                                pixel value with the mean value of that image and 
                                store it in the list (m*n long),do the same for remaining 
                                81 images in the directory, resulting in 81 lists.
                                use bisection algorithm for calculating the wavelength of the 
                                minimum value for the list , then calculate the maximum deviation from
                                the minimum point thus calculating the doppler shift , and calculating the 
                                line of sight velocity of chromosphere.
'''



intermediate_store_path = "E:\mean_correctd_x_y"
read_path = "E:\DATAMASTCA10052019\DATA_NOT_CORRECTED"
depth=81
file_wave = open("wavelength_array","rb")
wavelength_array = pickle.load(file_wave)
cluster_size = 10
mean_array = []
rows=1110
cols=1100
losimage = np.zeros((rows,cols))


counter=0
for file in os.listdir(read_path):
    hdul = fits.open(os.path.join(read_path, file))
    matrix = hdul[0].data
    rows, cols = matrix.shape
    reduced_matrix = ia.compress_matrix(matrix, cluster_size)
    #print(file)
    #plt.imshow(reduced_matrix,cmap="gray")
    #plt.show()
    mean_array.append(ia.mean_sinlge_mat(reduced_matrix))


retval,retval_y = ia.bisector(mean_array,wavelength_array,0.5)
x_val = np.mean(retval)
y_val = np.interp(x_val,wavelength_array,mean_array)
print(retval)
#print(y_val)
result = np.where(mean_array == np.min(mean_array))
print(wavelength_array[int(result[0])])
print("min",np.min(mean_array))
print("min",np.where(mean_array == np.min(mean_array)))
#plt.plot(wavelength_array,mean_array)
#plt.plot(x_val,y_val,'ro')
#plt.plot(retval[0], retval_y[0],'ro')
#plt.plot(retval[1], retval_y[1],'ro')
#plt.plot(retval[2], retval_y[2],'ro')
#plt.plot(retval[3], retval_y[3],'ro')
#plt.plot(retval[4], retval_y[4], 'ro')
#plt.plot(retval[5], retval_y[5], 'ro')
#plt.plot(retval[6], retval_y[6], 'ro')
#plt.show()


while True:
    pass


file2 = open("mean_array_not_corrrected","wb")

pickle.dump(mean_array,file2)
file2.close()

itr_r=itr_c=0
while itr_r < rows:
    while itr_c < cols:
        point_list = []
        for file in os.listdir(read_path):
            hdul = fits.open(os.path.join(read_path, file))
            matrix = hdul[0].data
            rows, cols = matrix.shape
            reduced_matrix = ia.compress_matrix(matrix, cluster_size)
            point_list.append(reduced_matrix[itr_r,itr_c])
        plt.plot(wavelength_array,point_list)
        plt.show()
        itr_c+=1
    itr_r+=1




