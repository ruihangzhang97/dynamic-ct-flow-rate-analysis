
###################################################################
# SETUP
###################################################################

# importing relevant libraries
import numpy as np
import matplotlib as plt
from pydicom import dcmread
import matplotlib.pyplot as plt
from PIL import Image as im 

# importing in file path to view DICOM
path = "/Users/felicialiu/Desktop/summer2022/coding/Cardiac 1.0 CE - 3/"
path1 = "/Users/felicialiu/Desktop/summer2022/coding/Cardiac 1.0 CE - 3/IM-0001-0001-0001.dcm"
path2 = "/Users/felicialiu/Desktop/summer2022/coding/Cardiac 1.0 CE - 3/IM-0001-0020-0001.dcm"
# print(path)

print("#######################################################\n\n")



###################################################################
# FUNCTION DEFINITIONS
###################################################################


# acquisition times are already sorted
def print_acqtimes(path, end_range):
    filler = 0
    count = 1
    prev = 0.0

    for i in range(1, end_range + 1):
        num_in = str(filler) + str(count)
        name = path + "IM-0001-00" + num_in + "-0001.dcm"

        count += 1
        if count == 10:
            filler += 1
            count = 0
        
        x = dcmread(name) 
        cur = float(x.AcquisitionDateTime)
        store = cur - prev
        print(num_in, ":", cur, "diff:", store)
        prev = cur

# same as acquisistion times without the date beforehand
def print_contimes(path, end_range):
    filler = 0
    count = 1
    prev = 0.0

    for i in range(1, end_range + 1):
        num_in = str(filler) + str(count)
        name = path + "IM-0001-00" + num_in + "-0001.dcm"

        count += 1
        if count == 10:
            filler += 1
            count = 0
        
        x = dcmread(name) 
        cur = x.ContentTime
        print(num_in, ":", cur)
        
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

# used as helper to save and pixel array as grayscale
def saver(array, num_in):
    # printer(array)
    # hist(array)
    name = '/Users/felicialiu/Desktop/summer2022/coding/photos/pic' + num_in + '.png'
    plt.imsave(name, array, cmap=plt.cm.gray)
    
# takes the images list and makes the gif and saves it to the folder
def save_the_gif(typee, choose, end_range, images):
    name = typee + "gif_slice_" + str(choose) + "_with_" + str(end_range) + "_entries"
    namee = '/Users/felicialiu/Desktop/summer2022/coding/gifs/' + name + '.gif'
    images[0].save(namee, 
            save_all = True, 
            append_images = images[1:], 
            duration=60, 
            loop=0)


# function takes in lots of sets of dicoms, extracts a certain slice
# called "choose" and combines into one giant new matrix accross time
# takes whole folder of dicoms
def axial_timemaker(path, choose, end_range, save_pics = False, gif_make = False):
    print("AXIAL: extracting slice", choose, "from", end_range, "DICOMS...")

    holder = []
    pre_gif = []

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
        # pixels = HU_corr(x.pixel_array[choose])
        holder.append(pixels)

        # CALL the saving function here to simulataneously keep all the photos
        # pixels is the actual array to save
        # name is what the png will be called
        if save_pics == True:
            saver(pixels, num_in)

        # if we want to make a gif, pixel arrays will be saved as images
        if gif_make == True:
            data = im.fromarray(pixels)
            pre_gif.append(data)
        
    # at the end, make holder an array
    holder = np.array(holder) 
    pre_gif = np.array(pre_gif) 

    if gif_make == True:
        save_the_gif("AXIAL", choose, end_range, pre_gif)
        
    
    print("done axial time extraction.\n")
    return holder

# takes in fullset of axial and makes new pixel array fully coronal
def coronal_timemaker(path, choose, end_range, save_pics = False, gif_make = False):
    print("CORONAL: extracting slice", choose, "from", end_range, "DICOMS...")

    holder = []
    pre_gif = []

    hundreds = 0
    tens = 0
    ones = 1

    # for each DICOM in the whole folder, we're going to:
    # first re-arrange it's orientation, then cut a slice out
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

        put_in = extend(x.pixel_array)

        xx = np.rot90(put_in, k=1, axes=(0, 1))
        pixels = xx[choose]
        # pixels = HU_corr(x.pixel_array[choose])
        holder.append(pixels)

        # CALL the saving function here to simulataneously keep all the photos
        # pixels is the actual array to save
        # name is what the png will be called
        if save_pics == True:
            saver(pixels, num_in)

        # if we want to make a gif, pixel arrays will be saved as images
        if gif_make == True:
            data = im.fromarray(pixels)
            pre_gif.append(data)
        
    # at the end, make holder an array
    holder = np.array(holder) 
    pre_gif = np.array(pre_gif) 

    if gif_make == True:
        save_the_gif("CORONAL", choose, end_range, pre_gif)
    
    print("done coronal time extraction.\n")
    return holder

# takes in fullset of axial and makes new pixel array fully sagital
def sagital_timemaker(path, choose, end_range, save_pics = False, gif_make = False):
    print("SAGITAL: extracting slice", choose, "from", end_range, "DICOMS...")

    holder = []
    pre_gif = []

    hundreds = 0
    tens = 0
    ones = 1

    # for each DICOM in the whole folder, we're going to:
    # first re-arrange it's orientation, then cut a slice out
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
        put_in = extend(x.pixel_array)
        xx = np.rot90(put_in, k=1, axes=(0, 1))
        xxx = np.rot90(xx, k=1, axes=(0, 2))
        pixels = xxx[choose]
        # pixels = HU_corr(x.pixel_array[choose])
        holder.append(pixels)

        # CALL the saving function here to simulataneously keep all the photos
        # pixels is the actual array to save
        # name is what the png will be called
        if save_pics == True:
            saver(pixels, num_in)

        # if we want to make a gif, pixel arrays will be saved as images
        if gif_make == True:
            data = im.fromarray(pixels)
            pre_gif.append(data)
        
    # at the end, make holder an array
    holder = np.array(holder) 
    pre_gif = np.array(pre_gif) 

    if gif_make == True:
        save_the_gif("SAGITAL", choose, end_range, pre_gif)
    
    print("done sagital time extraction.\n")
    return holder

# coronal and sagital need their dimension elongated by 4
# input array, output elongated array
def extend(array):

    new = []
    for i in array:
        new.append(i)
        new.append(i)
        new.append(i)

    new = np.array(new)
    return new



###################################################################
# MAIN
###################################################################

# print_contimes(path, 20)

# main folder path with the dicom photos
path = "/Users/felicialiu/Desktop/summer2022/coding/Cardiac 1.0 CE - 3/"

# # AXIAL:
# ax_choose = 100           # height slice variable [0 - 159] 160 depth
# ax_end_range = 30       # time variable [2 - 151]
# time_pics = axial_timemaker(path, ax_choose, ax_end_range, save_pics = False, gif_make = True)
# # print_stack(time_pics)

# CORONAL:
cor_choose = 100            # height slice variable [0 - 511] 512 long
cor_end_range = 70        # time variable [2 - 151]
time_pics = coronal_timemaker(path, cor_choose, cor_end_range, save_pics = False, gif_make = True)
# print_stack(time_pics)

# # SAGITAL:
# sag_choose = 287            # height slice variable [0 - 511] 512 long
# sag_end_range = 151         # time variable [2 - 151]
# time_pics = sagital_timemaker(path, sag_choose, sag_end_range, save_pics = False, gif_make = True)
# # print_stack(time_pics)





# # making a bunch of gifs
# L = [3, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 157]
# for i in L:
#     ax_choose = i          # height slice variable [0 - 159] 160 depth
#     ax_end_range = 5       # time variable [2 - 151]
#     time_pics = axial_timemaker(path, ax_choose, ax_end_range, save_pics = False, gif_make = True)









###################################################################
# CLEAN UP
###################################################################

# just to mark the end of the script
print("\n\ndone")

print("\n\n#######################################################")