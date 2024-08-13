'''
the goal of this is to bring in coupled pairs of CT images and
their respective segmented models, if we can have both as arrays
of coordinates, then overlaying one with other is the intensity 
of the cross section, then we can see if there's some major trend
over the course of a cardiac cycle.
'''

###################################################################
# SETUP
###################################################################

# importing relevant libraries
import numpy as np
import matplotlib as plt
from pydicom import dcmread
import matplotlib.pyplot as plt
import time

from PIL import Image

###################################################################
# FUNCTION DEFINITIONS
###################################################################

# function reads pngs given filname
def load_png(path):
    pass



###################################################################
# INPUTS
###################################################################



# importing in all file paths
path = "/Users/felicialiu/Desktop/summer2022/coding/test_intensity/"
path2 = "/Users/felicialiu/Desktop/summer2022/coding/test_intensity2/"

# print(path)



print("#######################################################\n\n")



###################################################################
# MAIN
###################################################################
L = [10, 30, 40, 60, 70, 80, 90]

# check all photos have been sized correctly
def check_sizes(path):
    for i in range(1, 8):
        p = path + "C" + str(i) +".png"
        im = Image.open(p, "r")
        pix_val = list(im.getdata())
        print(len(pix_val))
    print("\n")
    return

# for the small set of C images, input path, output vector of intensities
def compute_sums(folder_path):

    vec = []
    for i in range(1, 8):
        name = "C" + str(i) +".png"
        p = folder_path + name
        im = Image.open(p).convert('L')
        pix_values = list(im.getdata())
        
        vec.append(sum(pix_values)/len(pix_values))

        # print("name:", name)
        # print("\tlength:", len(pix_values))
        # print("\tsum:", sum(pix_values)/len(pix_values))
    
    print("\n")
    return vec

# check_sizes(path)
vec1 = compute_sums(path)
print(vec1)


# check_sizes(path2)
vec2 = compute_sums(path2)
print(vec2)

 
plt.plot(L, vec1, label='lower')
plt.plot(L, vec2, label='higher')
plt.xlabel("time through the cardiac cycle [%]")
plt.ylabel("intensity of the pixels")
plt.title("Intensity Time Curve (TIC) - Case 11 CABG Project")
plt.legend()
plt.xlim(0, 100)
plt.ylim(0, 255)
plt.show()


###################################################################
# CLEAN UP
###################################################################

# just to mark the end of the script
print("\n\ndone")

print("\n\n#######################################################")
