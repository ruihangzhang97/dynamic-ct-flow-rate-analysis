
###################################################################
# SETUP
###################################################################

# importing relevant libraries
from itertools import count
from re import I
import numpy as np
import matplotlib as plt
from pydicom import dcmread
import matplotlib.pyplot as plt
from PIL import Image as im 
import os
import time

# importing in file path to view DICOM
path = "/Users/felicialiu/Desktop/summer2022/coding/Cardiac 1.0 CE - 3/"
path1 = "/Users/felicialiu/Desktop/summer2022/coding/Cardiac 1.0 CE - 3/IM-0001-0001-0001.dcm"
path2 = "/Users/felicialiu/Desktop/summer2022/coding/Cardiac 1.0 CE - 3/IM-0001-0020-0001.dcm"
# print(path)

print("#######################################################\n\n")



###################################################################
# FUNCTION DEFINITIONS
###################################################################
 
# helper function to show DICOM
def printer(what):
    print("showing DICOM...")
    plt.imshow(what, cmap=plt.cm.gray)
    plt.show()
    return

# helper function to show the frequency of HU values per slice
def hist(what):
    print("showing frequency over HU values...")
    plt.hist(what.flatten(), bins=80, color='c')
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    plt.show()

# fixes the HU units for one slice
def HU_corr(pixelgrid):
    # print("making colour correction...")
    for i in range(len(pixelgrid)):
        for j in range(len(pixelgrid[i])):
            if pixelgrid[i][j] < -1000:
                pixelgrid [i][j] = -1000
            else:
                pass
    # print("colour correction done.\n")
    return pixelgrid

# uses the printer function on all matrices in a giant array
def print_stack(pics):
    for i in pics:
        printer(i)
    return

# inputting whole set of dynamic dicoms, gives stacks of matrices
# gif_pics that are better resolution
def axial_gifspawner(path, choose, end_range):
    print("AXIAL: gif spawning slice", choose, "from", end_range, "DICOMS...")

    holder = []

    hundreds = 0
    tens = 0
    ones = 1

    # for each DICOM in the whole folder, we're going to extract one slice
    for i in range(end_range):

        # just prepares the name of the file
        num_in = str(hundreds) + str(tens) + str(ones)
        name = path + "IM-0001-0" + num_in + "-0001.dcm"
        ones += 1
        if ones == 10:
            tens += 1
            ones = 0
            if tens == 10:
                hundreds += 1
                tens = 0
        
        # now we fix the colour and append the new grid to a premade list
        x = dcmread(name) 
        pixels = x.pixel_array[choose]
        
        # take the pixel array and quickly save and reextract as gif
        data = im.fromarray(pixels)
        namee = '/Users/felicialiu/Desktop/summer2022/coding/gifs/eh.gif'
        data.save(namee)
        wow = im.open(namee)
        hold = np.array(wow)
        os.remove(namee)

        holder.append(hold)
        
    # at the end, make holder an array
    holder = np.array(holder) 

    print("done axial gif extraction.\n")
    return holder

# takes a stack of matrices and averages out intensity outputs
# one mean (average) representation
def flattener(gif_pics):
    print("flattening gif stack...")
    tic = time.perf_counter()
    flattened_array = []

    for i in range(len(gif_pics[0])):

        mini = []
        for j in range(len(gif_pics[0][0])):
            
            sum = 0
            for frame in range(len(gif_pics)):
                sum = sum + gif_pics[frame][i][j]
            value = int(sum/len(gif_pics))

            mini.append(value)
        flattened_array.append(mini)
        

    flattened_array = np.array(flattened_array)
    toc = time.perf_counter()
    print("done flattening.\n")
    tictoc(tic, toc)
    return flattened_array

# used as helper to save and pixel array as grayscale
def saver(array, use):
    # printer(array)
    # hist(array)
    name = '/Users/felicialiu/Desktop/summer2022/coding/photos/pic' + use + '.png'
    plt.imsave(name, array, cmap=plt.cm.gray)
  
# runs a gradient function on an array
def get_grad_pixels(pixels):
    print("making a gradient...")
    grad_pixels = []
    for i in range(len(pixels)):
        mini = []
        for j in range(len(pixels[i])):
            
            # always start with this
            mid = pixels[i][j]
            LL = pixels[i][j]
            TL = pixels[i][j]
            TT = pixels[i][j]
            TR = pixels[i][j]
            RR = pixels[i][j]
            BR = pixels[i][j]
            BB = pixels[i][j]
            BL = pixels[i][j]
            # top left corner
            if i == 0 and j == 0:
                RR = pixels[i][j+1]
                BR = pixels[i+1][j+1]
                BB = pixels[i+1][j]
            # top right corner
            elif i == 0 and j == (len(pixels[i]) - 1):
                LL = pixels[i][j-1]
                BB = pixels[i+1][j]
                BL = pixels[i+1][j-1]
            # bot right corner
            elif i == (len(pixels) - 1) and j == (len(pixels[i]) - 1):
                LL = pixels[i][j-1]
                TL = pixels[i-1][j-1]
                TT = pixels[i-1][j]
            # bot left corner
            elif i == (len(pixels) - 1) and j == 0:
                TT = pixels[i-1][j]
                TR = pixels[i-1][j+1]
                RR = pixels[i][j+1]
            # top edge
            elif i == 0:
                LL = pixels[i][j-1]
                RR = pixels[i][j+1]
                BR = pixels[i+1][j+1]
                BB = pixels[i+1][j]
                BL = pixels[i+1][j-1]
            # left edge
            elif j == 0:
                TT = pixels[i-1][j]
                TR = pixels[i-1][j+1]
                RR = pixels[i][j+1]
                BR = pixels[i+1][j+1]
                BB = pixels[i+1][j]
            # bot edge
            elif i == (len(pixels) - 1):
                LL = pixels[i][j-1]
                TL = pixels[i-1][j-1]
                TT = pixels[i-1][j]
                TR = pixels[i-1][j+1]
                RR = pixels[i][j+1]
            # right edge
            elif j == (len(pixels[i]) - 1):
                LL = pixels[i][j-1]
                TL = pixels[i-1][j-1]
                TT = pixels[i-1][j]
                BB = pixels[i+1][j]
                BL = pixels[i+1][j-1]
            # it's a fully surrounded piece
            else:
                LL = pixels[i][j-1]
                TL = pixels[i-1][j-1]
                TT = pixels[i-1][j]
                TR = pixels[i-1][j+1]
                RR = pixels[i][j+1]
                BR = pixels[i+1][j+1]
                BB = pixels[i+1][j]
                BL = pixels[i+1][j-1]

            sum = abs(LL-mid) + abs(TL-mid) + abs(TT-mid) + abs(TR-mid) + abs(RR-mid) + abs(BR-mid) + abs(BB-mid) + abs(BL-mid)
            mini.append(5*sum)

        grad_pixels.append(mini)

    grad_pixels = np.array(grad_pixels)
    print("gradient done.\n")
    return grad_pixels

# checking the time of certain functions, manually place tic-toc
def tictoc(tic, toc):
    runtime = toc-tic
    print("runtime:", runtime, "\n")
# tic = time.perf_counter()
# toc = time.perf_counter()
# tictoc(tic, toc)

# opens iso and gets ready for intensity extraction
def prep_iso(iso_path):

    # opens and stores iso in that slices folder so we can compare to all sums
    image = im.open(iso_path).convert('L')
    iso = np.array(image)

    return iso

# takens in all gif pics and iso to make vector of intensity values
def intensity_extraction(gif_pics, iso):
    print("extracting intensity vector...")
    tic = time.perf_counter()
    vec = []
    for frame in gif_pics:
        
        sum = 0
        count = 0

        for i in range(len(frame)):
            for j in range(len(frame[i])):

                if iso[i][j] > 100:
                    sum = sum + frame[i][j]
                    count += 1
                else:
                    pass
        
        vec.append(round(sum/count))
    print("intensities extracted.\n")
    toc = time.perf_counter()
    tictoc(tic, toc)
    return vec


###################################################################
# TRYING STUFF
###################################################################

# # main folder path with the dicom photos
# path = "/Users/felicialiu/Desktop/summer2022/coding/Cardiac 1.0 CE - 3/"

# AXIAL: MUST HAVE EXISTING GIF
ax_choose = 95          # height slice variable [0 - 159] 160 depth
ax_end_range = 5      # time variable [2 - 151]
gif_pics = axial_gifspawner(path, ax_choose, ax_end_range)

# STEP #1 -> generates the flat pic and gradient image labbeled
# TYPEflat_slice_CHOOSE_with_ENDRANGE_entries.png
# TYPEgrad_slice_CHOOSE_with_ENDRANGE_entries.png
# flat = flattener(gif_pics)
# printer(flat)
# saver(flat, "flat3")
# flatter = get_grad_pixels(flat)
# printer(flatter)
# saver(flatter, "flat4")

iso_path = "/Users/felicialiu/Desktop/summer2022/coding/photos/iso.png"
iso = prep_iso(iso_path)
TIC_vec = intensity_extraction(gif_pics, iso)
print(TIC_vec)
print(len(TIC_vec))


# takes in intensity vectors and graphs with respect to 0.1 s time
def TIC_maker(TIC_vec, TYPE, choose):
    x = []
    # first make the x vector:
    for i in range(len(TIC_vec)):
        x.append( float("%.1f" % (i * (0.1))))
    
    title = TYPE + ": Intensity Time Curve (TIC) of Slice " + str(choose) + " - Case 1 CABG Project"
    
    plt.plot(x, TIC_vec, label='TIC')
    plt.xlabel("time of perfusion acquisition [s]")
    plt.ylabel("intensity of the pixels [grayscale]")
    plt.title(title)
    plt.legend()
    plt.xlim(0, 17)
    plt.ylim(0, 255)
    plt.show()
    
    return x


x = TIC_maker(TIC_vec, "AXIAL", ax_choose)




###################################################################
# CLEAN UP
###################################################################

# just to mark the end of the script
print("\n\ndone")

print("\n\n#######################################################")
