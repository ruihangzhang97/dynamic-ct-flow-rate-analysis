

###################################################################
# SETUP
###################################################################

# importing relevant libraries
import os
import numpy as np
from PIL import Image as im 
import matplotlib.pyplot as plt
import random
import math
import time


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

# helper function to show DICOM
# takes in pixel array
def printer(what):
    print("showing picture...\n")
    plt.imshow(what, cmap=plt.cm.gray)
    plt.show()
    return

# helper function to show the frequency of HU values per slice
# takes in pixel array
def hist(what):
    print("showing frequency over HU values...")
    plt.hist(what.flatten(), bins=80, color='c')
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    plt.show()

# uses the printer function on all matrices in a giant array
# takes in vector of pixel arrays
def print_stack(pics):
    for i in pics:
        printer(i)
    return

# for all images in the folder named 1..., put them into a vector
# vector contains pixel arrays
def images_to_vecarray(folder):
    num_images = len(os.listdir(folder))
    vec =[]
    for i in range(1, num_images):
        path = folder + "/" + str(i) + ".png"
        vec.append(image_to_pixels(path))
    vec = np.array(vec)
    return vec

# takes in a folder path to an image and spits out the pixel array
def image_to_pixels(path):
    image_pixels = im.open(path)
    # image_pixels = im.open(path).convert('L')
    image_pixels = np.array(image_pixels)
    return image_pixels

# makes unlinked seperate matrix
def pixel_copy(pixels):
    new = []
    for i in range(len(pixels)):
        mini = []
        for j in range(len(pixels[i])):
            mini.append(pixels[i][j])
        new.append(mini)
    new = np.array(new)
    return new

# for each of the locations in list locations, spits out new pixel
# array where those locations pixels are colours red
# basically shows which points in location and what they correspond to
def selected_shower(pixels, locations, centres, showw = True, typee = "normal"):
    
    if typee == "normal":
        new_array = pixel_copy(pixels)
    else: # working with grad here
        new_array = change1to3array(pixels)
    
    for i in locations:
        # new_array[i[0]][i[1]] = [255, 0, 0, 255]
        new_array[i[0]][i[1]] = [255, 0, 0]
    
    for i in centres:
        # new_array[i[0]][i[1]] = [0, 255, 0, 255]
        new_array[i[0]][i[1]] = [0, 255, 0]

    if showw == True:
        printer(new_array)

    return new_array

# return false, if the point doesn't exist on the grid
def its_ongrid(pixels, point):
    # 0 <= point[0] <= len(pixels)-1]
    if point[0] > len(pixels)-1:
        return False
    elif point[0] < 0:
        return False
    elif point[1] > len(pixels[0])-1:
        return False
    elif point[1] < 0:
        return False
    return True

# takes the pixels vec and makes the gif and saves it to the folder
def save_the_gif(name, axis, level, pixels, speed, saveable = True):

    if saveable == False:
        return
    
    images = []
    for i in pixels:
        # img_2d = i.astype(float)
        # img_2d_scaled = (np.maximum(img_2d,0) / img_2d.max()) * 255.0
        img_2d_scaled = np.uint8(i)
        data = im.fromarray(img_2d_scaled)
        # data = im.fromarray(i)
        images.append(data)
    images = np.array(images)

    namee = '/Users/felicialiu/Desktop/summer2022/coding/gifs/' + name + axis + str(level) +'.gif'
    images[0].save(namee, 
            save_all = True, 
            append_images = images[1:], 
            duration=speed, 
            loop=0)

# takes in a path to a gif and extracts its photos into pixel array
def extract_gif(path):
    pixel_array = []
    image = im.open(path)

    for frame in range(1,image.n_frames):
        image.seek(frame)
        n = np.array(image)
        pixel_array.append(n)
    
    pixel_array = np.array(pixel_array)
    return pixel_array

# colour mapping to make the right units
# redefines pixels to be within 0 to 255
def my_colourmap(pixels):

    low = np.amin(pixels)
    high = np.amax(pixels)
    # low = -300
    thres = 0
    # high = 1000

    blank = []
    for i in range(len(pixels)):
        blank.append([0]*len(pixels[0]))
    blank = np.array(blank)

    for i in range(len(pixels)):
        for j in range(len(pixels[i])):
            value = (((pixels[i][j])-low)/(high-low))*255
            if value < thres:
                value = 0
            if value >255:
                value = 255
            blank[i][j] = round(value)
    
    return blank


###################################################################
# MAIN FUNCTION DEFINITIONS
###################################################################
 
# takes in one pixelgrid and one centre point and colours in around it
# centre point is in the bolus
def initialise_frame(pixels, centrepoint, centres, thres, typee = "normal"):
    
    # will append locations within the bolus to this list
    locations = [centrepoint]
    queue = [centrepoint]

    ask = 0

    # for each point in the queue, we want to add to the queue all 
    # the points around it that fall within the threhold too
    while len(queue) > 0:
        point = queue[0]
        queue.remove(point)
        addees = find_all_around(pixels, point, thres, typee)
        new_locs = new_locations(locations, addees)
        # print(len(new_locs))

        for i in new_locs:
            locations.append(i)

        queue = remove_doubles(queue, new_locs)
        ask += 1

        if ask == 500:
            ask = 0
            selected_shower(pixels, locations, centres, True, typee)

    return locations

# identify if a point has strayed too far away compared to others
def too_far():
    pass

# look at the 8 locations around the point, if exceeding threshold
# it's valuable to keep, to add to the queue, else we discard
def find_all_around(pixels, point, thres, typee = "normal"):
    row = point[0]
    col = point[1]

    # open set holds all the points around
    open_set = []
    open_set.append([row, col-1]) #LL
    open_set.append([row-1, col-1]) #TL
    open_set.append([row-1, col]) #TT
    open_set.append([row-1, col+1]) #TR
    open_set.append([row, col+1]) #RR
    open_set.append([row+1, col+1]) #BR
    open_set.append([row+1, col]) #BB
    open_set.append([row+1, col-1]) #BL

    random.shuffle(open_set)

    pass_on = []
    # now they must pass two conditions:
    # EXIST and surpass the threshold
    if typee == "normal":
        for ele in open_set:
            if its_ongrid(pixels, ele): # gives true if point on grid
                if pixels[ele[0]][ele[1]][0] >= thres:
                    pass_on.append(ele)
    else: # if we have the grad version
        for ele in open_set:
            if its_ongrid(pixels, ele): # gives true if point on grid
                if pixels[ele[0]][ele[1]] <= thres:
                    pass_on.append(ele)
    return pass_on

# combine list 1 and 2 and return masterlist
def remove_doubles(list1, list2):
    for ele in list1:
        if ele in list2:
            list2.remove(ele)
    
    list3 = []
    for i in list1:
        list3.append(i)
    for i in list2:
        list3.append(i)
    return list3

# remove elements in potential already in locations
def new_locations(locations, potentials):
    for ele in locations:
        if ele in potentials:
            potentials.remove(ele)
    return potentials

# compare locations to new frame to determine which still stand
# returns list of locations in new frame still bright enough
def still_locations(next_frame, init_locations, thres):
    new_locations = []
    for ele in init_locations:
        if next_frame[ele[0]][ele[1]][0] >= thres:
            new_locations.append(ele)
    return new_locations

# with a bunch of locations, it starts with those as locations and 
# throws them in the queue to find all new neighbours
def re_colour(pixels, new_cen, size, thres):
    
    # will append locations within the bolus to this list
    locations = [new_cen]
    queue = [new_cen]

    count = 0

    # for each point in the queue, we want to add to the queue all 
    # the points around it that fall within the threhold too
    while len(queue) > 0 and count < size:
        point = queue[0]
        queue.remove(point)
        addees = find_all_around(pixels, point, thres)
        new_locs = new_locations(locations, addees)

        for i in new_locs:
            locations.append(i)
            count += 1

        queue = remove_doubles(queue, new_locs)
        # selected_shower(pixels, locations)

    return locations
    
# estimated the centre of a bunnch of locations, by find the average
def find_centre(locations, centres):
    
    x = []
    y = []

    if len(locations) == 0:
        print("ERROR, nothing in locations!")
        return centres[-1]
    
    for i in range(len(locations)):
        x.append(locations[i][0])
        y.append(locations[i][1])
    
    xcen = round((max(x) + min(x))/2)
    ycen = round((max(y) + min(y))/2)

    return [xcen, ycen]

# looks at the last two centres and predicts the trajectory
def new_centre_tracer(centres):
    
    woah = []
    # for i = zero (x) and one (y)
    for i in range(2):
        one = centres[-2][i]
        two = centres[-1][i]
        move = two - one
        woah.append(two+move)
    return woah


def dist(one, two):
    x = two[0]-one[0]
    y = two[1]-one[1]
    return np.sqrt(x**2 + y**2)


def crop_pics(test_pics, ymin, ymax, xmin, xmax):

    main = []
    for w in range(len(test_pics)):

        array = []
        for i in range(ymin, ymax):
            array.append(test_pics[w][i][xmin:xmax])
        array = np.array(array)
        main.append(array)
    main = np.array(main)
    return main


def whole(test_pics, init_centre, threshold):
    
    reds = []
    centres = [init_centre]
    print(test_pics[0][init_centre[0]][init_centre[1]])


    # FOR THE FIRST FRAME LETS COLOUR IT IN
    init_frame = test_pics[0]
    init_locations = initialise_frame(init_frame, init_centre, centres, threshold)
    centres.append(find_centre(init_locations, centres))
    printer(init_frame)
    holder = selected_shower(init_frame,init_locations, centres, False)
    reds.append(holder)

    far = 0

    # MOVING INTO THE NEXT FRAME, WE'LL SEED WITH GIVEN LOCATIONS
    for i in range(1, len(test_pics)):

        print("starting", i, "frame")
        next_frame = test_pics[i]

        if far <= 2:
            size = round(1.1*len(init_locations))
        else:
            size = round(1*len(init_locations))

        trans_locations = still_locations(next_frame, init_locations, threshold)
        centres.append(find_centre(trans_locations, centres))
        selected_shower(next_frame,trans_locations, centres, False)

        new_cen = new_centre_tracer(centres)
        centres.append(new_cen)
        
        new_locations = re_colour(next_frame, new_cen, size, threshold)
        centres.append(find_centre(new_locations, centres))
        holder = selected_shower(next_frame,new_locations, centres, False)
        reds.append(holder)

        far = math.ceil(dist(centres[-1], centres[-3]))
        
        # set up for next cycle
        init_locations = new_locations
    
    # this is to make all the gifs
    reds = np.array(reds)
    save_the_gif(str(len(test_pics[0])),"redsss", threshold, reds, 80)
    save_the_gif(str(len(test_pics[0])),"test_caseee", threshold, test_pics, 80)

    # save_the_gif(str(len(test_pics[0]))+"reds", par_dir, reds)
    # save_the_gif(str(len(test_pics[0]))+"test_case", par_dir, test_pics)
    return



###################################################################
# MAIN
###################################################################

par_dir = "/Users/felicialiu/Desktop/summer2022/coding/movingbolus_test"
pictures_folder = "/Users/felicialiu/Desktop/summer2022/coding/movingbolus_test/test100"



# # FUNCTION DEFINITIONS
# test_pics = images_to_vecarray(pictures_folder)
# init_centre = [28, 40] # indices are all "go down go right"
# threshold = 50 #pixel_intensity -> work to grad?
# # print_stack(test_pics)
# whole(test_pics, init_centre, threshold)



# print("hi")
# a_gif = "/Users/felicialiu/Desktop/summer2022/coding/gifs/AXIALgif_slice_100_with_30_entries.gif"
# test_pics = extract_gif(a_gif)
# test_pics = crop_pics(test_pics, 280, 350, 50, 150)
# # print_stack(test_pics[0:1])
# init_centre = [40, 60] # indices are all "go down go right"
# threshold = 50
# whole(test_pics, init_centre, threshold)


# runs a gradient function on an array
def get_grad_pixels(pixels):
    
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
            mini.append(sum)

        grad_pixels.append(mini)

    grad_pixels = np.array(grad_pixels)
    
    return grad_pixels

# runs gradient for a whole 3D pixel grid
def get_grad3D(pixel_grid):
    print("making a gradient grid...")
    tic = time.perf_counter()
    new = []
    for i in pixel_grid:
        pixels = i
        new.append(my_colourmap(get_grad_pixels(pixels)))
    toc = time.perf_counter()
    print("gradient done.")
    tictoc(tic, toc)
    return np.array(new)

# let's me choose which pixels to zero out
# takes in a 3D stack
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
    print("pixels zeroed.")
    return pixelgrid

# pixels are given in [1,1,1] or just 1 format
# takes in the weird vec to make single values
# assumes all 3 have the same values though
def change3to1(pixelgrid):
    pixel1D = []
    print("converting [ , , ] to single...")
    tic = time.perf_counter()
    for i in range(len(pixelgrid)):
        sheet = []
        for j in range(len(pixelgrid[i])):
            line = []
            for k in range(len(pixelgrid[i][j])):
                value = int(pixelgrid[i][j][k][0])
                line.append(value)
            sheet.append(line)
        pixel1D.append(sheet)
    toc = time.perf_counter()
    print("storage conversion complete.")
    tictoc(tic, toc)
    return np.array(pixel1D)

# pixels are given in [1,1,1] or just 1 format
# takes in the weird vec to make single values
# assumes all 3 have the same values though
def change1to3(pixelgrid):
    pixel1D = []
    print("converting single to [ , , ]...")
    tic = time.perf_counter()
    for i in range(len(pixelgrid)):
        sheet = []
        for j in range(len(pixelgrid[i])):
            line = []
            for k in range(len(pixelgrid[i][j])):
                value = int(pixelgrid[i][j][k])
                line.append([value, value, value])
                # line.append([250, 0, 0])
            sheet.append(line)
        pixel1D.append(sheet)
    toc = time.perf_counter()
    print("storage conversion complete.")
    tictoc(tic, toc)
    return np.array(pixel1D)


def change1to3array(pixels):
    sheet = []
    for j in range(len(pixels)):
        line = []
        for k in range(len(pixels[j])):
            value = int(pixels[j][k])
            line.append([value, value, value])
            # line.append([250, 0, 0])
        sheet.append(line)
    return np.array(sheet)


def change3to3(pixelgrid):
    pixel1D = []
    print("converting [ , , ] to [ , , ]...")
    tic = time.perf_counter()
    for i in range(len(pixelgrid)):
        sheet = []
        for j in range(len(pixelgrid[i])):
            line = []
            for k in range(len(pixelgrid[i][j])):
                value = int(pixelgrid[i][j][k][0])
                line.append([value, value, value])
            sheet.append(line)
        pixel1D.append(sheet)
    toc = time.perf_counter()
    print("storage conversion complete.")
    tictoc(tic, toc)
    return np.array(pixel1D)



# let's try something with the gradient function now
# to define what's in and what's out of a shape
path = "/Users/felicialiu/Desktop/summer2022/coding/gifs/AORTAslice20frames11.gif"
here = "/Users/felicialiu/Desktop/summer2022/coding/gifs"
test_pics = extract_gif(path)
test_pics = test_pics[40:50]
test_pics = change3to3(test_pics)
# grad_pics = get_grad3D(shapecorrected_pics)
# pic = test_pics[6]
# printer(pic)

# first lets try the old algorithum on this:
init_centre = [125, 125] # indices are all "go down go right"
threshold = 120
whole(test_pics, init_centre, threshold)


# centres = [init_centre]
# init_frame = get_grad_pixels(test_pics[0])
# printer(init_frame)
# print(init_frame)
# init_locations = initialise_frame(init_frame, init_centre, centres, threshold, "grad")








# shapecorrected_pics = change3to1(test_pics)
# grad_pics = get_grad3D(shapecorrected_pics)
# print_stack(grad_pics)
# save_the_gif("grad_pics", "20frames", 1, grad_pics, 80)

###################################################################
# CLEAN UP
###################################################################

# just to mark the end of the script
print("\n\ndone")

print("\n\n#######################################################")