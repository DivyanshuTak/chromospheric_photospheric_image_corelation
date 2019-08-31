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
read_path = "E:\DATAMASTCA10052019\CORRECTED_X_Y"
depth=81
file_wave = open("wavelength_array","rb")
wavelength_array = pickle.load(file_wave)



def list_of_points(read_path):
    cluster_size = 2
    data_list = []
    counter=0
    for file in os.listdir(read_path):
        data_list = []
        hdul = fits.open(os.path.join(read_path, file))
        matrix = hdul[0].data
        rows, cols = matrix.shape
        reduced_matrix = ia.compress_matrix(matrix, cluster_size)
        mean_val = ia.mean_sinlge_mat(reduced_matrix)
        max_val = np.max(reduced_matrix)
        data_list = (reduced_matrix.flatten()/mean_val)
        #for a in range(rows):
        #    for b in range(cols):
        #        ratio = matrix[a, b] / mean_val
        #        data_list.append(ratio)
        name = str(counter)
        counter+=1
        #print(len(data_list))
        file = open(os.path.join(intermediate_store_path,name),"wb")
        pickle.dump(data_list,file)
        file.close()
    return 1


def bisection(iplist,wavelength_list):
    skip = int(len(iplist)/81)+1
    itr=0
    a=0
    final_list = []#np.zeros((skip,depth))
    temp_pixel_list = []
    cut_list = []
    while itr<depth:
        temp_pixel_list = []
        a=itr
        while a <= len(iplist):
            temp_pixel_list.append(iplist[a])
            a += skip
        final_list.append(temp_pixel_list)#[:,itr] = temp_pixel_list
        itr+=1
    return final_list
    '''
    while a < depth:
        cut_list = list[a:]
        while itr <= len(cut_list):
            temp_pixel_list.append(cut_list[itr])#temp_pixel_list.append(list[(itr*depth)+a])
            itr+=81
        final_list[:,a] = temp_pixel_list
        a+=1
    return final_list
    '''


def bisection_2(ippath):
    a=0
    while a<81:
        val_list = []
        for file in os.listdir(ippath):
            file2 = open(os.path.join(ippath,file), "rb")
            data = pickle.load(file2)
            val_list.append(data[a])
        plt.plot(wavelength_array,val_list)
        plt.show()
        a+=1
    return 1















val = bisection_2(intermediate_store_path)
print(val)
#out = list_of_points(read_path)
#print(out)
#final_list = bisection(list_of_points(read_path),wavelength_array)
#file2 = open("lineofsight","wb")
#pickle.dump(final_list,file2)
#file2.close()
#print(len(final_list[0]))

#for a in range(depth):
#    plt.plot(wavelength_array,final_list[a])
#    plt.show()












