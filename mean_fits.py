import numpy as np
from astropy.io import fits
import cv2
import os
import pickle

def mean_fits(path_to_file,loadfile):
    base_path = path_to_file
    total_files = len(os.listdir(base_path))
    print("the number of total fits images are:",total_files)
    print("\n")
    stack=np.zeros(shape=(2048,2060))
    for file in os.listdir(base_path):
        hdulist = fits.open(os.path.join(base_path,file))
        image = hdulist[0].data
        stack = np.add(stack , image)
    mean_stack = np.divide(stack,total_files)
    file = open(loadfile,'wb')
    try:
        pickle.dump(mean_stack, file)
        file.close()
        return 1
    except:
        print("error in dumping the files")
        return 0



if __name__ == "__main__":
    path = "E:\MASTDARK"
    print(mean_fits(path,'mean_dark'))
