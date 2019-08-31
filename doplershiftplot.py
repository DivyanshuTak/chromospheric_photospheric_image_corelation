import numpy as np
import cv2
import os
from astropy.io import fits
import pickle
import image_arithematic as ia
import matplotlib.pyplot as plt

file = open("lineofsight","rb")
data = pickle.load(file)
file.close()

print(len(data[0]))