import numpy as np
import cv2
import os
from astropy.io import fits
import pickle
import image_arithematic as ia
import matplotlib.pyplot as plt

#=======================================================================================================================================
#               maximum shift assumed is of 30 pixels
#               to change the caliberation for the shift just change the SHIFT_MAX value and the rest will be done automatically
#=======================================================================================================================================
SHIFT_MAX = 8
BATCH_SIZE = 20
TOTAL = (2*SHIFT_MAX) + 1
BASE_PATH = "E:\DATAMASTCA10052019\DATA"
X_START = 330
Y_START = 400
DELTAX = 1110
DELTAY = 1100
SUM_OF_DIFFERENCE = np.zeros(TOTAL)
SUM_OF_DIFFERENCE_Y = np.zeros(TOTAL)
SHIFT_VAL = np.zeros(TOTAL)
INDEX_ARRAY = np.zeros(TOTAL)
REF_IMAGE = np.zeros((DELTAX,DELTAY))
BATCH_BUFFER = np.zeros((DELTAX,DELTAY,BATCH_SIZE))
count=0
count2=0
#TOTAL_SET_CORRECTED = np.zeros((DELTAX,DELTAY,81))
BUFFER_SET_CORRECTED = np.zeros((DELTAX,DELTAY,20))
#TOTAL_SET_UNCORRECTED = np.zeros((DELTAX,DELTAY,81))
DESTINATION_PATH = "E:\DATAMASTCA10052019\CORRECTED_X_Y"
list_of_pixel_shift_x = []
list_of_pixel_shift_y = []
#===========================================================================================================================
#=====================================================================================================================================
#                                       SUM OF DIFFERENCE CALCULATOR FUNCTION
def shift_and_calculate_x(ref_mat , current_mat , SHIFT ):
    array =  np.zeros(TOTAL)
    sliced_val_curr = ia.slicer(current_mat, X_START, Y_START, DELTAX, DELTAY)
    sliced_val_ref = ia.slicer(ref_mat, X_START, Y_START, DELTAX, DELTAY)
    a = 1
    while a <= SHIFT:                                                      #left
        array[a-1] = ia.mean_sinlge_mat(ia.diff_mat(sliced_val_ref, ia.shift_matrix(sliced_val_curr, a, 1, 0, 0)))
        a += 1


    array[SHIFT] = ia.mean_sinlge_mat(ia.diff_mat(sliced_val_ref, sliced_val_curr))

    a = 1
    while a <= SHIFT:                                                           #right
        array[a + SHIFT ] = ia.mean_sinlge_mat(ia.diff_mat(sliced_val_ref, ia.shift_matrix(sliced_val_curr, a, 1, 0, 1)))
        a += 1

    return array

def shift_and_calculate_y(ref_mat , current_mat , SHIFT ):
    array =  np.zeros(TOTAL)
    sliced_val_curr = ia.slicer(current_mat, X_START, Y_START, DELTAX, DELTAY)
    sliced_val_ref = ia.slicer(ref_mat, X_START, Y_START, DELTAX, DELTAY)

    a = 1
    while a <= SHIFT :                                                                      # down

        array[a - 1] = ia.mean_sinlge_mat(ia.diff_mat(sliced_val_ref, ia.shift_matrix(sliced_val_curr, a, 0, 1, 0)))
        a += 1

    array[SHIFT] = ia.mean_sinlge_mat(ia.diff_mat(sliced_val_ref, sliced_val_curr))

    a = 1
    while a <= SHIFT:                                                                               #top

        array[a + SHIFT] = ia.mean_sinlge_mat(ia.diff_mat(sliced_val_ref, ia.shift_matrix(sliced_val_curr, a, 0, 1, 1)))
        a += 1

    return array



#===========================================================================================================================
#===================================================order of shift is left  - 0 - right ====================================================
#
#=========================


for file in os.listdir(BASE_PATH):
    hdul = fits.open(os.path.join(BASE_PATH, file))
    count += 1
    BATCH_BUFFER[:, :, (count - 1)] = ia.slicer(hdul[0].data,X_START,Y_START,DELTAX,DELTAY)

    if (count == BATCH_SIZE):
        REF_IMAGE = BATCH_BUFFER[:,:,0]
        BUFFER_SET_CORRECTED[:,:,0] = REF_IMAGE
        #hdu = fits.PrimaryHDU(REF_IMAGE)
        #hdulist = fits.HDUList([hdu])
        #hdulist.writeto(os.path.join(DESTINATION_PATH,'0.fits'))
        t=1
        while t<(BATCH_SIZE):
            SUM_OF_DIFFERENCE = shift_and_calculate_x(REF_IMAGE,BATCH_BUFFER[:,:,t],SHIFT_MAX)
            SUM_OF_DIFFERENCE_Y = shift_and_calculate_y(REF_IMAGE,BATCH_BUFFER[:,:,t],SHIFT_MAX)
            touple_x = np.where(SUM_OF_DIFFERENCE == np.amin(SUM_OF_DIFFERENCE))
            index_x = int(touple_x[0])
            touple_y = np.where(SUM_OF_DIFFERENCE_Y == np.amin(SUM_OF_DIFFERENCE_Y))
            index_y = int(touple_y[0])
            temp_matrix = np.zeros((DELTAX,DELTAY))
            temp_matrix2 = np.zeros((DELTAX, DELTAY))
            if (index_x < SHIFT_MAX):
                list_of_pixel_shift_x.append(index_x)
            elif (index_x >= SHIFT_MAX):
                list_of_pixel_shift_x.append(index_x-SHIFT_MAX)
            if (index_y < SHIFT_MAX):
                list_of_pixel_shift_y.append(index_y)
            elif (index_y >= SHIFT_MAX):
                list_of_pixel_shift_y.append(index_y-SHIFT_MAX)
            #print(index_x,index_y)

            if index_x < SHIFT_MAX:  # left shifted
                # count2 += 1
                shift_pixels_x = index_x

                # new_name = str(count2) + '.fits'
                temp_matrix = ia.shift_matrix(BATCH_BUFFER[:, :, t], shift_pixels_x, 1, 0, 0)
                if index_y < SHIFT_MAX:  # down shifted
                    shift_pixels_y = index_y
                    temp_matrix2 = ia.shift_matrix(temp_matrix, shift_pixels_y, 0, 1, 0)
                elif index_y > SHIFT_MAX:
                    shift_pixels_y = index_y - SHIFT_MAX
                    temp_matrix2 = ia.shift_matrix(temp_matrix, shift_pixels_y, 0, 1, 1)
                elif index_y == SHIFT_MAX:
                    shift_pixels_y = 0
                    temp_matrix2 = ia.shift_matrix(temp_matrix, shift_pixels_y, 0, 1, 0)

                # qw

            elif index_x == SHIFT_MAX:  # no shift
                # count2 += 1
                # new_name = str(count2) + '.fits'
                shift_pixels_x = 0
                temp_matrix = ia.shift_matrix(BATCH_BUFFER[:, :, t], shift_pixels_x, 1, 0, 0)
                if index_y < SHIFT_MAX:  # down shifted
                    shift_pixels_y = index_y
                    temp_matrix2 = ia.shift_matrix(temp_matrix, shift_pixels_y, 0, 1, 0)

                elif index_y > SHIFT_MAX:
                    shift_pixels_y = index_y - SHIFT_MAX
                    temp_matrix2 = ia.shift_matrix(temp_matrix, shift_pixels_y, 0, 1, 1)

                elif index_y == SHIFT_MAX:
                    shift_pixels_y = 0
                    temp_matrix2 = ia.shift_matrix(temp_matrix, shift_pixels_y, 0, 1, 0)

                # hdu = fits.PrimaryHDU(BATCH_BUFFER[:,:,t])#(temp_matrix2)
                # hdulist = fits.HDUList([hdu])
                # hdulist.writeto(os.path.join(DESTINATION_PATH, new_name))

            elif index_x > SHIFT_MAX:  # right shift
                # count2 += 1
                shift_pixels_x = index_x - SHIFT_MAX
                temp_matrix = ia.shift_matrix(BATCH_BUFFER[:, :, t], shift_pixels_x, 1, 0, 1)
                if index_y < SHIFT_MAX:  # down shifted
                    shift_pixels_y = index_y
                    temp_matrix2 = ia.shift_matrix(temp_matrix, shift_pixels_y, 0, 1, 0)

                elif index_y > SHIFT_MAX:
                    shift_pixels_y = index_y - SHIFT_MAX
                    temp_matrix2 = ia.shift_matrix(temp_matrix, shift_pixels_y, 0, 1, 1)

                elif index_y == SHIFT_MAX:
                    shift_pixels_y = 0
                    temp_matrix2 = ia.shift_matrix(temp_matrix, shift_pixels_y, 0, 1, 0)

                # new_name = str(count2) + '.fits'

            BUFFER_SET_CORRECTED[:, :, t] = temp_matrix2









                #hdu = fits.PrimaryHDU(BATCH_BUFFER[:,:,t])#(temp_matrix2)
                #hdulist = fits.HDUList([hdu])
                #hdulist.writeto(os.path.join(DESTINATION_PATH, new_name))
            t+=1
        #new_name = str(count2) + '.fits'
        #hdu = fits.PrimaryHDU(ia.mean_of_volume(BUFFER_SET_CORRECTED))  # (temp_matrix2)
        #hdulist = fits.HDUList([hdu])
        #hdulist.writeto(os.path.join(DESTINATION_PATH, new_name))
        #count2 += 1
        #BATCH_BUFFER[:,:,:] = 0
        print("batch completed")
        count = 0



file_x = open('shift_pixels_x','wb')
pickle.dump(list_of_pixel_shift_x,file_x)
file_x.close()
file_y = open('shift_pixels_y','wb')
pickle.dump(list_of_pixel_shift_y,file_y)
file_y.close()

print("done !!")






























































