###################################################################
# SETUP
###################################################################


# importing relevant libraries
import numpy as np
import matplotlib as plt
from pydicom import dcmread
import matplotlib.pyplot as plt
import time

import numpy as np
from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = [9, 8]
plt.rcParams["figure.autolayout"] = True

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')



###################################################################
# INPUTS
###################################################################


# importing in file path to view DICOM
path = "/Users/felicialiu/Desktop/summer2022/coding/test_patient"
# print(path)

beginner = 580  # where the first file is
ender = 620     # where the last file is

# body parts
body_parts_dict = { "air" : [-2048, -500], #-1000
                    "lungs" : [-500, -100], #-500
                    "fat" : [-100, -50], 
                    "water" : [-50, 10], #0
                    "CSF" : [10, 20], #15
                    "kidney" : [20, 30], #30
                    "blood" : [30, 45], 
                    "muscle" : [10, 40],
                    "grey_matter" : [37, 45],
                    "white_matter" :[20, 30],
                    "liver" : [40, 60],
                    "soft_tissue" : [100, 300],
                    "bone" : [700, 3000]}
chosen_part = "lungs" # choose a body part to look at here


print("#######################################################\n\n")



###################################################################
# FUNCTION DEFINITIONS
###################################################################


# THEIRS: kaggle challenge, source code from online
def load_scan(path):
    print("Loading scan", path)
    #slices = [dcmread(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
 
    if slices[0].ImagePositionPatient[2] == slices[1].ImagePositionPatient[2]:
        sec_num = 2;
        while slices[0].ImagePositionPatient[2] == slices[sec_num].ImagePositionPatient[2]:
            sec_num = sec_num+1;
        slice_num = int(len(slices) / sec_num)
        slices.sort(key = lambda x:float(x.InstanceNumber))
        slices = slices[0:slice_num]
        slices.sort(key = lambda x:float(x.ImagePositionPatient[2]))
 
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices # list of DICOM 

# MINE: the point of this is the create list of all 
# the read dicom folders for each slice of the patient
def my_load_patient(path, begin, end):

    # initialising things
    print("loading scans from:", path)
    slices = []

    for s in range(begin, end):
        slice = dcmread(path + "/I" + str(s))
        slices.append(slice)


    print("patient slices created\n")
    return slices 

# THEIRS: convert to hounsfield units (HU) per se
def get_pixels_hu(slices):
    print("get pixels converted to HU")
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)
    for slice_number in range(len(slices)):        
        intercept = slices[slice_number].RescaleIntercept ## DICOM metadata attribute call
        slope = slices[slice_number].RescaleSlope         ## DICOM metadata attribute call
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
 
    case_pixels = np.array(image, dtype=np.int16)
    pixel_spacing = np.array([slices[0].SliceThickness, slices[0].PixelSpacing[0], slices[0].PixelSpacing[1]], dtype=np.float32)
    return case_pixels, pixel_spacing

# MINE: this doesn't really do anything because all 
# the slopes are 1 and all intercepts are 0, so it's 
# not doing any sort of correction...
def my_get_hu(slices):
    
    print("converting pixel numbers to HU ...")
    
    # image = np.stack([s.pixel_array for s in slices])
    # # Convert to int16 (from sometimes int16), 
    # # should be possible as values should always be low enough (<32k)
    # image = image.astype(np.int16)
    
    for slice_number in range(len(slices)):        
        intercept = slices[slice_number].RescaleIntercept ## DICOM metadata attribute call
        slope = slices[slice_number].RescaleSlope         ## DICOM metadata attribute call
        
        print("intercept", intercept, "slope", slope)
        # if slope != 1:
        #     image[slice_number] = slope * image[slice_number].astype(np.float64)
        #     image[slice_number] = image[slice_number].astype(np.int16)
            
        # image[slice_number] += np.int16(intercept)
 
    # case_pixels = np.array(image, dtype=np.int16)
    # pixel_spacing = np.array([slices[0].SliceThickness, slices[0].PixelSpacing[0], slices[0].PixelSpacing[1]], dtype=np.float32)
    # return case_pixels, pixel_spacing
    return

# helper function to show DICOM
def printer(chart):
    print("showing DICOM...")
    plt.imshow(pixels[chart], cmap=plt.cm.gray)
    plt.show()
    return

# helper function to show the frequency of HU values per slice
def hist(chart):
    print("showing frequency over HU values...")
    plt.hist(pixels[chart].flatten(), bins=80, color='c')
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    plt.show()

# returns just a 3D thing of pixels
# we want to create a vector of matrices, essentially making a 3D 
# thing of all the pixels greyscales that makes to the full shape
def extract_pixels(slices):
    print("extracting pixels...")
    pixels = []
    for slice in slices:
        pixels.append(slice.pixel_array)
    
    pixels2 = np.array(pixels)
    print("pixels extracted.")
    return pixels2

# theoretically this should handle converting the gray scale units
# to HU scale, but I'm not super sure at this moment what currently
# are the grayscale values I'm looking at, if they were HU, then in 
# 3D plotting, we could choose how much of certain densities we want
# to have loaded, or just choosing one material by looking at ranges
def HU_corr(pixelgrid):
    print("making colour correction...")
    for i in range(len(pixelgrid)):
        for j in range(len(pixelgrid[i])):
            for k in range(len(pixelgrid[i][j])):

                if pixelgrid[i][j][k] < -1000:
                    pixelgrid [i][j][k] = -1000
                else:
                    pass
    print("HU values corrected\n")
    return pixelgrid

# let's me choose which pixels to zero out
def correction(pixelgrid, low_thres, high_thres):
    print("zeroing chosen pixels...")
    for i in range(len(pixelgrid)):
        for j in range(len(pixelgrid[i])):
            for k in range(len(pixelgrid[i][j])):

                if (pixelgrid[i][j][k] >= low_thres) and (pixelgrid[i][j][k] <= high_thres):
                    # it's a pixel we want to see
                    pass
                else:
                    # it's not of interest for viewing, so we zero it
                    pixelgrid [i][j][k] = 0
    print("pixels zeroed, just body part.")
    return pixelgrid

# THEIRS: to fix spacing issues that I think ours didnt have
def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
    return image, new_spacing

# THEIRS: we want to see what all the slices look like together? 
def plot_3d(image, threshold = -300):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    
    verts, faces = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()

# MINE:
def plotter3D(image, threshold = -300):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    
    verts, faces = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()

# checking the time of certain functions, manually place tic-toc
def tictoc(tic, toc):
    runtime = toc-tic
    print("runtime:", runtime)


###################################################################
# TRYING STUFF
###################################################################


# all of the patients images are now loaded in
patient = my_load_patient(path, beginner, ender)

# then we extract the pixels and try to clean up
# pixels = HU_corr(extract_pixels(patient))
pixels = extract_pixels(patient)

print(pixels)
# body_part will store a corrected version of only pixels with 
# values in that range, everything else is zeroed
# also times how long that takes
# body_thres =  body_parts_dict[chosen_part] 
# tic = time.perf_counter()
# body_part = correction(pixels, body_thres[0], body_thres[1])
# print(body_part)
# toc = time.perf_counter()
# tictoc(tic, toc)
# scrolling through every pixel in a stack of 40 takes about 20 seconds
# when we run 1000 files, it should take 500 seconds, which is 8 minutes


# plots to remaining pixels, aka the body_part chosen
# tic = time.perf_counter()
# ax.voxels(body_part)
# toc = time.perf_counter()
# tictoc(tic, toc)
# plt.show()
# running the voxels plotting function took around 270 seconds for 40
# scaled to 1000 pictures, it'd be 6750secs -> 112 mins -> 2 hours


# just wanted to check the size of the data set
# print("z:", len(pixels), "x:",len(pixels[0]), "y:",len(pixels[0][0]), "\n")


# visualisation
# printer(20) # the number is the slice index, which right now runs 0-39
# hist(20)


# haven't figured out what this does yet, it doesn't work for me...
# pix_resampled, spacing = resample(pixels, patient, [1,1,1])
# plot_3d(pixels, 400)


###################################################################
# CLEAN UP
###################################################################

# just to mark the end of the script
print("done")

print("\n\n#######################################################")