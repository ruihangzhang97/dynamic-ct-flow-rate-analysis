'''
i input how many colours i need and it makes it between red and purple
'''


###################################################################
# SETUP
###################################################################

import os
import numpy as np
from PIL import Image as im 
import matplotlib.pyplot as plt
import random
import math
import time
from pydicom import dcmread


print("#######################################################\n\n")



###################################################################
# HELPER FUNCTION DEFINITIONS
###################################################################

# checking the time of certain functions, manually place tic-toc
def tictoc(tic, toc):
    runtime = toc-tic
    print("runtime:", runtime, "\n")
# tic = time.perf_counter()
# toc = time.perf_counter()
# tictoc(tic, toc)

# simply plots a vector on y with arbitry x
def plainplot(vec):
    print("running plain plotting...")
    
    x = []
    # first make the x vector:
    for i in range(len(vec)):
        x.append(i)
    
    title = "PLAIN PLOT"
    
    plt.scatter(x, vec)
    plt.xlabel("arbitrary x")
    plt.ylabel("the given y")
    plt.yticks(np.arange(min(vec), max(vec)+1, 5.0))
    plt.title(title)
    plt.show()
    return

# helper function to show DICOM
# takes in pixel array
def printer(woo):

    # just fixing the dimension so always square
    what = []
    for i in range(len(woo[0])):
        what.append(woo[0])

    print("showing picture...\n")
    plt.imshow(what, cmap=plt.cm.gray)
    plt.show()
    return




###################################################################
# FUNCTION DEFINITIONS - 
###################################################################







###################################################################
# MAIN
###################################################################
aaa = time.perf_counter()



def return_colours(num):
    r = 255
    g = 0
    b = 0
    br = round(num/3-0.3) 
    gb = round(num/3-0.3)
    rg = num - br - gb
    list = []

    # first pass : red to green
    for n in range(rg):
        list.append([r, g, b])
        sub = round(255/rg)
        sub2 = round((255/rg)/2)
        # r = max(0, r-sub2)
        g = min(255, g+sub)
    
    r = 0
    # first pass : green to blue
    for n in range(gb):
        list.append([r, g, b])
        sub = round(255/gb)
        sub2 = round((255/gb)/2)
        g = max(0, g-sub+10)
        b = min(255, b+sub2+10)
    
    g = 0
    b = 255
    # first pass : blue to purple
    for n in range(br):
        list.append([r, g, b])
        sub = round(150/br)
        sub2 = round((150/br)/2)
        b = max(0, b-sub2)
        r = min(255, r+sub)

    printer([list])

    return list
    
def convert_rgbtohex(L):
    hex = []
    for three in L:
        use = (three[0], three[1], three[2])
        hex.append('#%02x%02x%02x' % use)
    return hex


num = 16
rgb = return_colours(num)
hex = convert_rgbtohex(rgb)
print(hex)



bbb = time.perf_counter()
###################################################################
# CLEAN UP
###################################################################

# just to mark the end of the script

print("\n\nWHOLE RUNTIME:")
tictoc(aaa, bbb)
print("done")

print("\n\n#######################################################")