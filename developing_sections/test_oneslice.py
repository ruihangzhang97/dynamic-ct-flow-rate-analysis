###################################################################
# SETUP
###################################################################

# importing relevant libraries
import numpy as np
import matplotlib as plt
from pydicom import dcmread
import matplotlib.pyplot as plt

# importing in file path to view DICOM
path = "/Users/felicialiu/Desktop/summer2022/coding/test_patient/I620"
# print(path)

print("#######################################################\n\n")



###################################################################
# FUNCTION DEFINITIONS
###################################################################


x = dcmread(path)       # class object "FileDataSet" has attributes
print(x)               # gives a bunch of info organised with headings, the class object
print(dir(x))          # just gives the types of things you can call in a giant list
#print(x.PixelSpacing)  #[0.473, 0.473] spacing between pixels in the images
# print(x.pixel_array)  # gives a giant matrix 
print("#cols:",len(x.pixel_array[0]), "#rows:", len(x.pixel_array))



# fixes the HU units
def HU_corr(pixelgrid):
    print("making colour correction...")
    for i in range(len(pixelgrid)):
        for j in range(len(pixelgrid[i])):
            if pixelgrid[i][j] < -1000:
                pixelgrid [i][j] = -1000
            else:
                pass
    print("colour correction done.\n")
    return pixelgrid

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

# let's me choose which pixels to zero out
def correction(pixelgrid, low_thres, high_thres):
    print("zeroing some pixels...")
    for i in range(len(pixelgrid)):
        for j in range(len(pixelgrid[i])):
            if (pixelgrid[i][j] >= low_thres) and (pixelgrid[i][j] <= high_thres):
                # it's a pixel we want to see
                pixelgrid [i][j] = 1
                #pass
            else:
                # it's not of interest for viewing, so we zero it
                pixelgrid [i][j] = 0
    print("done zeroing pixels.\n")
    return pixelgrid

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



###################################################################
# TRYING STUFF
###################################################################


# plotting the DIMCON pixel array as a plot
# pixels = HU_corr(x.pixel_array)

# printer(pixels)
# hist(pixels)

# so it looks like the DICOM, it's only in a circle hence the edges 
# of the array are all -2048, which I'm guessing is super black

# let's us extract the gradient of the dicom photo
# pixels_grad = get_grad_pixels(pixels)

# printer(pixels_grad)
# hist(pixels_grad)

# the correction will zone in on a range of how steep gradients 
# we want to look at, 4000-100000 are the super bright lines,
# the fainter ones we can see from around 1000-3000
# pixels_thres = correction(pixels_grad, 1000, 3000)

# printer(pixels_thres)
# hist(pixels_thres)





###################################################################
# CLEAN UP
###################################################################

# just to mark the end of the script
print("\n\ndone")

print("\n\n#######################################################")