import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2

file = open('intensity_array','rb')
intensity_array = pickle.load(file)
file.close()

file2 = open('wavelength_array','rb')
wavelenght_array = pickle.load(file2)
file2.close()

print(wavelenght_array)
print("=============================================================================================")
print(intensity_array)

plt.plot(wavelenght_array , intensity_array)
plt.show()



