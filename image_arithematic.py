import numpy as np
from astropy.io import fits
import math
import pickle
from matplotlib import pyplot as plt

def sum_mat(mat1,mat2):
    rows1,cols1 = mat1.shape
    rows2, cols2 = mat2.shape
    if ((rows1 != rows2) or (cols2 != cols1)):
        raise Exception("matrix dimensions are not equal")

    return_mat = np.zeros(shape=(rows1,cols1))
    return_mat = np.add(mat1,mat2)
    return  return_mat


def diff_mat(mat1, mat2):
    rows1, cols1 = mat1.shape
    rows2, cols2 = mat2.shape
    if ((rows1 != rows2) or (cols2 != cols1)):
        raise Exception("matrix dimensions are not equal")

    return_mat = np.zeros(shape=(rows1, cols1))
    return_mat = abs(mat1 - mat2)
    return return_mat

def mul_mat_elementwise(mat1,mat2):
    rows1, cols1 = mat1.shape
    rows2, cols2 = mat2.shape
    if ((rows1 != rows2) or (cols2 != cols1)):
        raise Exception("matrix dimensions are not equal")

    return_mat = np.zeros(shape=(rows1, cols1))
    return_mat = np.multiply(mat1, mat2)
    return return_mat

def mean_of_volume(volume):
    (rows,cols,number) = volume.shape
    temp = np.zeros((rows,cols))
    a=0
    while (a < number):
        temp = np.add(temp,volume[:,:,a])
        a += 1
    return (np.divide(temp , number))

def mean_sinlge_mat(matrix):
    return np.mean(matrix)


def intensty_of_matrix(image , x_max , y_max):
    section = image[0:x_max , 0:y_max]
    return ((np.mean(section))*(x_max + y_max))

def wavelength_of_fits(hdul):
    #hdul = fits.open(path)
    return (hdul[0].header['WAVEID'] - (hdul[0].header['FP1VOL']*hdul[0].header['FP1VTR']))                             # wavelength = centre - voltage*tuning_ratio

def distance_calculator(x1,y1,x2,y2):
    diff = abs(x1-x2)
    diff_y = abs(y1-y2)
    tot = ((diff*diff) + (diff_y*diff_y))
    return math.sqrt(tot)

def compress_matrix(input,reduction_ratio):
    rows,cols = input.shape
    temp_mat = np.zeros((int(rows/reduction_ratio),int(cols/reduction_ratio)))
    a=0
    a1=0
    b=0
    b1=0
    while a<rows:
        while b<cols:
            temp = mean_sinlge_mat(input[a:a+reduction_ratio,b:b+reduction_ratio])
            temp_mat[a1,b1] = temp
            b+=reduction_ratio
            b1+=1
        b=0
        b1=0
        a+=reduction_ratio
        a1 += 1
    return temp_mat




#=========================================================================================
# axis =1 se left rifht  +ve val se right and -ve val se left
# axis  = 0 se top bottom and +val is bottom and -val is top
#=========================================================================================
def shift_matrix(matrix , val , x_shift , y_shift , left_right):
    rows,cols = matrix.shape
    matrix2 = np.zeros((rows,cols))
    if ((x_shift==1)and(y_shift==0)and(left_right==1)):                                     # right
        matrix2 = np.roll(matrix,val,axis=1)
        return matrix2
    elif ((x_shift==1)and(y_shift==0)and(left_right==0)):                                   # left
        matrix2 = np.roll(matrix, -val, axis=1)
        return matrix2
    elif ((x_shift==0)and(y_shift==1)and(left_right==1)):                                     # y=1,left_right=1 = top
        matrix2 = np.roll(matrix, -val, axis=0)
        return matrix2
    elif ((x_shift==0)and(y_shift==1)and(left_right==0)):                                     # y=1,left_right=0 = down
        matrix2 = np.roll(matrix, val, axis=0)
        return matrix2

#    rows,cols = matrix.shape
#    shifted_mat = np.zeros((rows,cols))
#    if ((x_shift==1)and(y_shift==0)and(left_right==1)):                                         # right shift
#        shifted_mat[:,0:(val-1)] = matrix[:,(cols - (val-1)):(cols-1)]
#       shifted_mat[:,val:(cols-1)] = matrix[:,0:(cols-val)]
#   elif ((x_shift==1)and(y_shift==0)and(left_right==0)):                                       # left shift
#       shifted_mat[:, (cols - (val - 1)):(cols - 1)] = matrix[:, 0:(val-1)]
#        shifted_mat[:, 0:(cols - val)] = matrix[:, val:cols]
#    elif ((x_shift==0)and(y_shift==1)and(left_right==0)):                                       # top shift is analogous to left shift
#       shifted_mat[:, (rows - (val - 1)):(rows - 1)] = matrix[:, 0:(val - 1)]
#        shifted_mat[:, 0:(rows - val)] = matrix[:, val:rows]
#    elif ((x_shift==0)and(y_shift==1)and(left_right==1)):                                       # down shift is analogous to right shift
#        shifted_mat[:, (rows - (val - 1)):(rows - 1)] = matrix[:, 0:(val - 1)]
#        shifted_mat[:, 0:(rows - val)] = matrix[:, val:rows]
#    return shifted_mat

def slicer(matrix , x , y , deltax , deltay):
    rows,cols = matrix.shape
    #sliced  = np.zeros((deltax,deltay))
    sliced =  matrix[x:x+deltax , y:y+deltay]
    return sliced

def slicer_space_image(matrix , x , y , deltax , deltay):                                   # beacuase of change of origin the coordinates are shifted
    rows,cols = matrix.shape
    #sliced  = np.zeros((deltax,deltay))
    sliced =  matrix[x:x+deltax , y:y-deltay]
    return sliced



def elementwise_add(val1,val2):
    return val1+val2

def elementwise_sub(val1,val2):
    return abs(val1-val2)

def elementwise_mul(val1,val2):
    return val1*val2

def image_mean(image):
    rows,cols = image.shape
    return np.mean(image)



def cal_aspect_ratio(image,threshold):
    return np.where(image < threshold)

def interpolate(y_val,x,y):
    for a in range(len(y)):
        if (abs(y[a] - y_val) < abs(y[a] - y[a+1])) and (abs(y[a+1] - y_val) < abs(y[a] - y[a+1])):#val>y[a] and y<y[a+1]:
            deltay = abs(y[a+1] - y[a])
            deltax = abs(x[a+1] - x[a])
            deltaydash = abs(y_val - y[a])
            ratio = deltaydash/deltay
            x_val = x[a] + ratio*deltax
            return x_val


def bisector(y_arr,x_arr,offset):
    bisector_array = []
    bisector_array_y = []
    if len(x_arr) == len(y_arr):
        pass
    else:
        raise Exception("the axis are not of same size")

    length = len(y_arr)
    ref_min_val = np.min(y_arr)
    result = np.where(y_arr == ref_min_val)
    ref_min_index = int(result[0])
    delta_limit_left = abs(x_arr[ref_min_index] - x_arr[ref_min_index-20])
    delta_limit_right = abs(x_arr[ref_min_index] - x_arr[ref_min_index + 20])
    #print(ref_min_index)
    #print(ref_min_val)
    a=20
    ratio=0.5
    diff=delta=0
    itr=5
    while a >= 1:

#=========================================================== BOUNDARY POINT ============================================
        val_l = y_arr[ref_min_index - a]
        x_val_l = x_arr[ref_min_index - a]

        #print("------------------------------")
        #print(val_l)
        #print(x_val_l)
        #print("-------------------------------")

        i=1
        itr=0
        while i:#diff >= delta:
            i-=1
            while val_l > y_arr[ref_min_index + itr]:
                itr+=1

            if val_l == y_arr[itr+ref_min_index]:
                x_val_r = x_arr[ref_min_index + itr]
                #print(x_val_r)
            elif val_l < y_arr[ref_min_index+itr]:
                diff = abs(val_l - y_arr[ref_min_index + itr -1])
                delta = abs(y_arr[itr+ref_min_index] - y_arr[itr+ref_min_index-1])
                ratio = diff/delta
                deltax = abs(x_arr[ref_min_index + itr-1] - x_arr[ref_min_index + itr ])
                x_val_r = x_arr[itr+ref_min_index-1] + ratio*deltax

            else:
                print("error")


        diff_check = abs(abs(np.mean([x_val_l,x_val_r]) - x_arr[ref_min_index]) - x_arr[ref_min_index])
        if (diff_check > delta_limit_left) and (diff_check < delta_limit_right):
            bisector_array.append(np.mean([x_val_l,x_val_r]))
            bisector_array_y.append(val_l)
#========================================================== INTERPOLATED POINT =========================================
        offset2 = abs(x_arr[ref_min_index - a] - x_arr[ref_min_index - a-1])*offset
        new_x = x_arr[ref_min_index - a ] - offset2
        offsety = abs(y_arr[ref_min_index - a] - y_arr[ref_min_index - a-1])*offset
        val_l_new = y_arr[ref_min_index - a ] - offsety
        #val_l_new = #np.interp(new_x,x_arr,y_arr)
        #print("============================")
        #print(new_x)
        #print(val_l_new)
        #print("============================="
        #print(val_l_new)
        itr=0
        #print(new_x,val_l_new)
        while val_l_new > y_arr[ref_min_index + itr]:
            itr += 1
        if val_l_new == y_arr[itr + ref_min_index]:
            new_x_val_r = x_arr[ref_min_index + itr]
        elif val_l_new < y_arr[ref_min_index + itr]:
            diff = abs(val_l_new - y_arr[ref_min_index + itr - 1])
            delta = abs(y_arr[itr + ref_min_index] - y_arr[itr + ref_min_index - 1])
            ratio = diff / delta
            #print(diff,delta)
            deltax = abs(x_arr[ref_min_index + itr - 1] - x_arr[ref_min_index + itr])
            new_x_val_r = x_arr[itr + ref_min_index - 1] + (ratio * deltax)
        else :
            print("error")
        diff_check = abs(np.mean([new_x,new_x_val_r]) - x_arr[ref_min_index])
        if (diff_check > delta_limit_left) and (diff_check < delta_limit_right):
            bisector_array.append(np.mean([new_x,new_x_val_r]))
            bisector_array_y.append(val_l_new)

        a-=1
    #print(len(bisector_array))
    return bisector_array,bisector_array_y


    #left_ext = y_arr[ref_min_index-offset]
    #itr = right_ext = y_arr[ref_min_index+offset]
    #l_val = left_ext
    #x_co_l = interpolate(l_val,x_arr,y_arr)
    #while itr >= ref_min_index and itr <= right_ext:
    #    if l_val



if __name__ == "__main__":
    arr1 = np.ones(shape=(100,100,10))
    arr2 = np.ones(shape=(4, 4))
    #array = np.array(([1,2,3],[4,2,3],[5,2,3]))
    #array2 = np.array([3,1,7,4,5,8,2])
    #array_2 = 10*np.ones((5,5,5))
    #a=10
    #print(distance_calculator(0,0,5,5))
    array1 = [10,9,8,7,6,5,4,3,2,1,2,3,4,5,6,7,8,9,10]
    array2 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,19,20]
    #bisector_array,val = bisector(array1,array2,4)
    #print(bisector_array)
    #print(val)
    #retval,retval_y = bisector(array1,array2,0.5)
    #print(len(retval),len(retval_y))
    #print(np.mean(retval))
    file = open("meanfortest","rb")
    array = pickle.load(file)
    file.close()
    file_wave = open("wavelength_array", "rb")
    wavelength_array = pickle.load(file_wave)
    file_wave.close()
    ret,rety =bisector(array,wavelength_array,0.5)
    mean_X = np.mean(ret)
    print(ret)
    print(rety)
    plt.plot(wavelength_array,array)
    plt.plot(ret,rety)


    plt.show()