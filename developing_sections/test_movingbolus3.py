'''
PROCESS:
- take in the set with the gif
- flatten it first to see where the aorta goes
- make an iso that just covers where the aorta moves to
- crop the picture using the size of that iso so it's easier to treat
- in the cropped image, apply a mask to everything out of the way using iso
- then extract locations as normal + colour in using selected shower
- those locations, convert to normal on the full picture
- then show those using selected shower
- and be able to extract intensities
'''

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

# makes unlinked full body of pixels
def pixel_grid_copy(pixel_grid):
    wow = []
    for slice in pixel_grid:
        wow.append(pixel_copy(slice))
    return np.array(wow)

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

# same deal ^
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

# same deal ^
def change4to1(pixels):
    print("converting [ , , , 255] to single...")
    tic = time.perf_counter()
    sheet = []
    for j in range(len(pixels)):
        line = []
        for k in range(len(pixels[j])):
            value = int(pixels[j][k][0])
            line.append(value)
        sheet.append(line)
    toc = time.perf_counter()
    print("storage conversion complete.")
    tictoc(tic, toc)
    return np.array(sheet)

# same deal ^
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

# same deal ^^
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

# takes the pixels vec and makes the gif and saves it to the folder
def save_the_gif(pixels, name, speed, here, saveable = True):

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

    namee = here + name +'.gif'
    images[0].save(namee, 
            save_all = True, 
            append_images = images[1:], 
            duration=speed, 
            loop=0)

# extract gif name assuming it came from the gif folder
def extract_gif_name(gif_path):
    new = gif_path.replace("/Users/felicialiu/Desktop/summer2022/coding/gifs/", "")
    new = new.replace(".gif", "")
    return new
    
# used as helper to save and pixel array as grayscale
def saver(array, use, here):
    name = here + "/" + use + '.png'
    plt.imsave(name, array, cmap=plt.cm.gray)
  
# extract only certain frames from test_pics
def time_clipper(L, test_pics):
    new = []
    for i in L:
        new.append(test_pics[i])
    return np.array(new)

# for each of the locations in list locations, spits out new pixel
# array where those locations pixels are colours red
# basically shows which points in location and what they correspond to
def selected_shower(pixels, locations, centres = [], showw = True, typee = "normal"):
    
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

# to change the front back of the pictures
def reverse_order(pixel_grid):
    new = []
    for i in range(len(pixel_grid)-1, -1, -1):
        new.append(pixel_grid[i])
    return np.array(new)





###################################################################
# FUNCTION DEFINITIONS - take in the set with the gif
###################################################################

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

# for each gif, create it's own directory to put all its stuff into
# and first, stick the gif in it!
def make_dir(parent_dir, gif_path):

    # start making folder with choose and end_range
    print("making directory...")
    print(gif_path)
    directory = extract_gif_name(gif_path)
    path = os.path.join(parent_dir, directory) 
    os.mkdir(path) 
    print("directory '% s' created.\n" % directory)

    here = parent_dir + "/" + directory + "/"

    return here







###################################################################
# FUNCTION DEFINITIONS - flatten it first to see where aorta goes
###################################################################

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
    
    print("done flattening.")
    tictoc(tic, toc)
    return flattened_array







###################################################################
# FUNCTION DEFINITIONS - crop the picture using iso
###################################################################

# makes the frames smaller to look more specifically
def crop_pics(test_pics, ymin, ymax, xmin, xmax):

    main = []
    for w in range(len(test_pics)):

        array = []
        for i in range(ymin, ymax+1):
            array.append(test_pics[w][i][xmin:xmax+1])
        array = np.array(array)
        main.append(array)
    main = np.array(main)
    return main

# estimated the centre of a bunch of locations, by find the average
def find_range(locations):
    
    x = []
    y = []

    if len(locations) == 0:
        print("ERROR, can't find range because nothing in locations!")
        return
    
    for i in range(len(locations)):
        x.append(locations[i][0])
        y.append(locations[i][1])
    
    x_range = [min(x)-2, max(x)+2]
    y_range = [min(y)-2, max(y)+2]

    return x_range, y_range

# takes a grid of white and black values and returns locs
def gridtolocs(pixels):
    locs = []
    new = pixel_copy(pixels)
    for i in range(len(pixels)):
        for j in range(len(pixels[i])):
            if pixels[i][j] > 30:
                locs.append([i, j])
                new[i][j] = 255
            else:
                new[i][j] = 0
    # printer(new)
    return np.array(locs)

# crop a picture based on white bounds and give mapping
def iso_based_cropper(pixel_grid, iso):
    # first get the locations
    locs = gridtolocs(iso)

    # then first the range of these locs
    x, y = find_range(locs)
    # print(x, y)

    # then crop pixels based on these new ranges
    crops = crop_pics(pixel_grid, x[0], x[1], y[0], y[1])
    
    # cropping iso too by the same margions
    sheet = []
    for i in range(x[0], x[1]+1, 1):
        line = []
        for j in range(y[0], y[1]+1, 1):
            line.append(iso[i][j])
        sheet.append(line)
    cropped_iso = np.array(sheet)

    # print(len(cropped_iso), len(cropped_iso[0]))
    # print(len(crops[0]), len(crops[0][0]))

    return cropped_iso, crops, [x, y]

def little_iso_range(iso):
    locs = gridtolocs(iso)
    x, y = find_range(locs)
    return x, y

# takes on pixel slice and for every point on it not in locations
# the pixel is dimmed by some amount
def grid_masker(slicee, iso, amount):
    new = pixel_copy(slicee)
    for i in range(len(slicee)):
        for j in range(len(slicee[i])):
            if iso[i][j] > 30:
                value = min(round(slicee[i][j][0] + 0), 255)
                new[i][j] = [value, value, value]
            else:
                value = max(round(slicee[i][j][0] - amount), 0)
                new[i][j] = [value, value, value]
    return new

# looks at the time pics and applys a darkening mask to all the pixels 
# outside of the locations by the given amount for the rest of time
def the_following_get_darker(pixel_grid, behind_this, iso, amount, chop_factor = 3, use = "full"):
    print("masking pictures", behind_this+1, "to", len(pixel_grid)-1, "by", amount, "intensity...")
    tic = time.perf_counter()

    # let's say i'm looking at behind_this = 2, then 3 to the end should get darker
    if use == "full":
        for w in range(behind_this+1, len(pixel_grid), 1):
            print("masking", w)
            pixel_grid[w] = grid_masker(pixel_grid[w], iso, amount)
    else: #nearest
        for w in range(behind_this+1, min(behind_this+chop_factor+1, len(pixel_grid)), 1):
            print("masking", w)
            pixel_grid[w] = grid_masker(pixel_grid[w], iso, amount)

    toc = time.perf_counter()
    print("masking done.")
    tictoc(tic, toc)
    return pixel_grid

# this will take iso and 
def iso_dealer(pixel_grid, here, amount):
    
    # first is to extract iso's pictures in single
    iso_path = here + "iso.png"
    iso = change4to1(image_to_pixels(iso_path))
    # printer(iso)

    # now we want to crop based on iso's size 
    ciso, cpixel_grid, rangee = iso_based_cropper(pixel_grid, iso)

    # now we can almost completely mask all pixels outside 
    # of iso on the cropped grid
    grid_in = change1to3(cpixel_grid)
    masked_pixel_grid = the_following_get_darker(grid_in, -1, ciso, amount)
    # print_stack(masked_pixel_grid)


    return masked_pixel_grid, rangee
    





###################################################################
# FUNCTION DEFINITIONS - extract locations as normal + colour in 
###################################################################

# selects all the pixels around a small square
def small_square_around(center, size, test_pics):
    locs = []
    for x in range(center[0]-round(size/2), center[0]+round(size/2), 1):
        for y in range(center[1]-round(size/2), center[1]+round(size/2), 1):
            locs.append([x, y])
    
    # the for each loc we find the intensity and report as sum
    ints = []
    thres = []
    for pixels in test_pics:
        sum = 0
        for loc in locs:
            sum = sum + pixels[loc[0]][loc[1]]
        sum = round(sum/len(locs))
        ints.append(sum)
        thres.append(sum-23)
    return np.array(locs), np.array(ints), np.array(thres)

# rescaling the thresholds
def thres_correcter(thres):
    max = np.max(thres)
    min = np.min(thres)

    newmin = min + 8
    newmax = max

    new_thres = []
    for t in thres:
        val = round(((t-min)/(max-min))*(newmax-newmin)+newmin)
        print(val)
        new_thres.append(val)

    return np.array(new_thres)

# you give it the stack, and it chooses the inital frame
# let's you choose the inital centre, and let's you determine 
# a moving threshold to accomadate the washing
def the_decider(test_pics, here):
    print("looking for the moving average...")

    # first thing is to flatten test_pics
    flat = flattener(test_pics)
    printer(flat)

    # now we want to extract intensity for a little 
    print("choose picture centre")
    down = input("down:  ")
    side = input("side:  ")
    init_centre = [int(down), int(side)]
    # init_centre = [40, 40]
    size = 10
    locs, ints, thres = small_square_around(init_centre, size, test_pics)
    thres = thres_correcter(thres)
    # centres = [init_centre]

    test_pics = change1to3(test_pics)
    
    # for pixels in test_pics:
    #     selected_shower(pixels, locs, centres)

    # the x vector
    x = []
    for i in range(len(ints)):
        x.append( float("%.1f" % (i * (0.1))))
    
    plt.plot(x, ints, color = "red", label = "pixel intensity")
    plt.plot(x, thres, color = "blue", label = "pixel threshold")
    plt.xlabel("time of perfusion acquisition [s]")
    plt.ylabel("intensity of the pixels [grayscale]")
    plt.title("ROUGH PREDICTION - time-intensity plot")
    plt.legend()
    plt.xlim(0, len(ints)/10+0.2)
    plt.ylim(0, 255)
    plt.grid()

    name = here + "roughints.png"
    plt.savefig(name)
    plt.show()

    # # while plotting, let's save the data to a txt file too!
    # name1 = "time of perfusion acquisition [s]"
    # thing1 = x
    # name2 = "intensity of the pixels [grayscale]"
    # thing2 = ints

    # name = here + "roughresults.txt"
    # textfile = open(name, "w")
    # textfile.write(name1 + "\n")
    # for element in thing1:
    #     textfile.write(str(element) + "\n")
    # textfile.write("\n")
    # textfile.write(name2 + "\n")
    # for element in thing2:
    #     textfile.write(str(element) + "\n")
    # textfile.close()

    peak = input("choose initial peak starter (decimal):  ")
    peak = float(peak)
    peak = int(peak*10)

    return thres, init_centre, ints, peak

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

        # if ask == 500:
        #     ask = 0
        #     selected_shower(pixels, locations, centres, True, typee)

    return locations

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

# takes on pixel slice and for every point on it not in locations
# the pixel is dimmed by some amount
def grid_masker2(slice, locations, amount):
    for i in range(len(slice)):
        for j in range(len(slice[i])):
            locloc = [i, j]
            if locloc not in locations: # meaning it's outside, we want to mask it
                value = max(round(slice[i][j][0] - amount), 0)
                slice[i][j] = [value, value, value]
            else: # it's in locations so it can get a little brighter
                value = min(round(slice[i][j][0] + amount), 255)
                slice[i][j] = [value, value, value]
    return slice

# looks at the time pics and applys a darkening mask to all the pixels 
# outside of the locations by the given amount for the rest of time
def the_following_get_darker2(pixel_grid, behind_this, locations, amount, chop_factor, use = "full"):
    print("masking pictures", behind_this+1, "to", len(pixel_grid)-1, "by", amount, "intensity...")
    tic = time.perf_counter()

    # let's say i'm looking at behind_this = 2, then 3 to the end should get darker
    if use == "full":
        for w in range(behind_this+1, len(pixel_grid), 1):
            print("masking", w)
            pixel_grid[w] = grid_masker2(pixel_grid[w], locations, amount)
    else: #nearest
        for w in range(behind_this+1, min(behind_this+chop_factor+1, len(pixel_grid)), 1):
            print("masking", w)
            pixel_grid[w] = grid_masker2(pixel_grid[w], locations, amount)

    toc = time.perf_counter()
    print("masking done.")
    tictoc(tic, toc)
    return pixel_grid

# there's a math function to do this but it returns distance
def dist(one, two):
    x = two[0]-one[0]
    y = two[1]-one[1]
    return np.sqrt(x**2 + y**2)

# does bolus tracing on every slide over time
def whole(test_picss, init_centre, thresholds, amount, here, ind = "", typee = "normal"):
    
    reds = []
    centres = [init_centre]
    LLOOCCS = []

    # have an editable copy to change
    test_pics = pixel_grid_copy(test_picss)
    
    # FOR THE FIRST FRAME LETS COLOUR IT IN
    init_frame = test_pics[0]
    init_locations = initialise_frame(init_frame, init_centre, centres, thresholds[0], typee)
    LLOOCCS.append(init_locations)
    centres.append(find_centre(init_locations, centres))
    printer(init_frame)
    holder = selected_shower(test_picss[0],init_locations, centres, True)
    reds.append(holder)

    chop_factor = 3
    use = "nearest"

    test_pics = the_following_get_darker2(test_pics, 0, init_locations, amount, chop_factor, use)
    far = 0

    # keep a running total of the sizes
    first = len(init_locations)
    sizes = [len(init_locations)]

    # MOVING INTO THE NEXT FRAME, WE'LL SEED WITH GIVEN LOCATIONS
    for i in range(1, len(test_picss)):

        print("starting", i, "frame with thres", thresholds[i])
        next_frame = test_pics[i]

        if far <= 2:
            size = min(round(1.03*len(init_locations)), round(1.1*first))
            sizes.append(size)
        else:
            size = max(round(0.97*len(init_locations)), round(0.9*first))
            sizes.append(size)

        trans_locations = still_locations(next_frame, init_locations, thresholds[i])
        centres.append(find_centre(trans_locations, centres))
        # selected_shower(test_picss[i],trans_locations, centres, False)

        new_cen = new_centre_tracer(centres)
        centres.append(new_cen)
        
        new_locations = re_colour(next_frame, new_cen, size, thresholds[i])
        LLOOCCS.append(new_locations)
        centres.append(find_centre(new_locations, centres))
        holder = selected_shower(test_picss[i],new_locations, centres, False)
        reds.append(holder)

        test_pics = the_following_get_darker2(test_pics, i, new_locations, amount, chop_factor, use)
        far = math.ceil(dist(centres[-1], centres[-3]))
        
        # set up for next cycle
        init_locations = new_locations

        print("\n\n")
    
    # this is to make all the gifs
    reds = np.array(reds)
    save_the_gif(reds, "REDrev"+ind, 80, here)
    save_the_gif(test_pics, "TESTrev"+ind, 80, here)
    
    return LLOOCCS, sizes

# depending on what the peak is, we need to cut the test_pics stack in half
# then run both sides seperately then stitch them back together
def whole_runner(test_pics, init_centre, thresholds, amount, here, peak):

    # the first half before and including peak is reversed
    # their outputs must also be reversed
    print("reversing and running front half...")
    locs1, sizes1 = whole(reverse_order(test_pics[:peak]), init_centre, reverse_order(thresholds[:peak]), amount, here, "1")
    locs1 = reverse_order(locs1)
    sizes1 = reverse_order(sizes1)

    # the second half is run as normal 
    # no reversals
    print("running second half...")
    locs2, sizes2 = whole(test_pics[peak:], init_centre, thresholds[peak:], amount, here, "2")

    # at the end we stitch both parts together
    print("stiching halfs back together...")
    locs_vec = []
    for i in locs1:
        locs_vec.append(i)
    for i in locs2:
        locs_vec.append(i)
    sizes = []
    for i in sizes1:
        sizes.append(i)
    for i in sizes2:
        sizes.append(i)
    
    print("done all frame extracting")
    return locs_vec, sizes

# useful to make an amounts vector that tips upwards on the ends
def decide_on_amounts(thres, amount, amount_in, amount_up):
    amounts = [amount]*len(thres)
    rollo = amount_up/amount_in

    count = 1
    for i in range(amount_in-1, -1, -1):
        amounts[i] = round(amounts[i] + count*rollo)
        count += 1
    count = 1
    for i in range(len(amounts)-amount_in, len(amounts), 1):
        amounts[i] = round(amounts[i] + count*rollo)
        count += 1

    return amounts

# does bolus tracing on every slide over time
# only difference is this one takes in a vector of amounts
def whole2(test_picss, init_centre, thresholds, amounts, here, ind = "", typee = "normal"):
    
    reds = []
    centres = [init_centre]
    LLOOCCS = []

    # have an editable copy to change
    test_pics = pixel_grid_copy(test_picss)
    
    # FOR THE FIRST FRAME LETS COLOUR IT IN
    init_frame = test_pics[0]
    init_locations = initialise_frame(init_frame, init_centre, centres, thresholds[0], typee)
    LLOOCCS.append(init_locations)
    centres.append(find_centre(init_locations, centres))
    # printer(init_frame)
    holder = selected_shower(test_picss[0],init_locations, centres, False)
    reds.append(holder)

    chop_factor = 3
    use = "nearest"

    test_pics = the_following_get_darker2(test_pics, 0, init_locations, amounts[0], chop_factor, use)
    far = 0

    # keep a running total of the sizes
    first = len(init_locations)
    sizes = [len(init_locations)]

    # MOVING INTO THE NEXT FRAME, WE'LL SEED WITH GIVEN LOCATIONS
    for i in range(1, len(test_picss)):

        print("starting", i, "frame with thres", thresholds[i], "and amount", amounts[i])
        next_frame = test_pics[i]

        if far <= 2:
            size = min(round(1.03*len(init_locations)), round(1.1*first))
            sizes.append(size)
        else:
            size = max(round(0.97*len(init_locations)), round(0.9*first))
            sizes.append(size)

        trans_locations = still_locations(next_frame, init_locations, thresholds[i])
        centres.append(find_centre(trans_locations, centres))
        # selected_shower(test_picss[i],trans_locations, centres, False)

        new_cen = new_centre_tracer(centres)
        centres.append(new_cen)
        
        new_locations = re_colour(next_frame, new_cen, size, thresholds[i])
        LLOOCCS.append(new_locations)
        centres.append(find_centre(new_locations, centres))
        holder = selected_shower(test_picss[i],new_locations, centres, False)
        reds.append(holder)

        test_pics = the_following_get_darker2(test_pics, i, new_locations, amounts[i], chop_factor, use)
        far = math.ceil(dist(centres[-1], centres[-3]))
        
        # set up for next cycle
        init_locations = new_locations

        print("\n\n")
    
    # this is to make all the gifs
    reds = np.array(reds)
    save_the_gif(reds, "REDrev"+ind, 80, here)
    save_the_gif(test_pics, "TESTrev"+ind, 80, here)
    
    return LLOOCCS, sizes

# depending on what the peak is, we need to cut the test_pics stack in half
# then run both sides seperately then stitch them back together
# this one calculates the amounts vector based on the thresholds vector
def whole_runner2(test_pics, init_centre, thresholds, amount, here, peak, amount_in, amount_up):

    # first it will make the full amounts vector based on the threshold vector
    amounts = decide_on_amounts(thresholds, amount, amount_in, amount_up)

    # the first half before and including peak is reversed
    # their outputs must also be reversed
    print("reversing and running front half...")
    locs1, sizes1 = whole2(reverse_order(test_pics[:peak]), init_centre, reverse_order(thresholds[:peak]), reverse_order(amounts[:peak]), here, "1")
    locs1 = reverse_order(locs1)
    sizes1 = reverse_order(sizes1)

    # the second half is run as normal 
    # no reversals
    print("running second half...")
    locs2, sizes2 = whole2(test_pics[peak:], init_centre, thresholds[peak:], amounts[peak:], here, "2")

    # at the end we stitch both parts together
    print("stiching halfs back together...")
    locs_vec = []
    for i in locs1:
        locs_vec.append(i)
    for i in locs2:
        locs_vec.append(i)
    sizes = []
    for i in sizes1:
        sizes.append(i)
    for i in sizes2:
        sizes.append(i)
    
    print("done all frame extracting")
    return locs_vec, sizes




###################################################################
# FUNCTION DEFINITIONS - those locations, convert to normal
###################################################################

# remap the locations to the real image
# rangee's first is down and second is side
# rangee = [[76, 161], [91, 171]] 
# [76, 91] is [0, 0]
# it'll fix the order too
def location_fixer(locs_vec, rangee):
    new_vec = []
    for locs in locs_vec:
        new_locs = []
        for loc in locs: # 
            down = loc[0] + rangee[0][0]
            side = loc[1] + rangee[1][0]
            new_locs.append([down, side])
        new_vec.append(new_locs)
    return new_vec

# shows and makes gif for fixed pixel locations
def real_locs_shower(real_locs, pixel_grid):
    holders = []
    print(len(real_locs), len(pixel_grid))
    for i in range(len(pixel_grid)):
        print(i)
        holder = selected_shower(pixel_grid[i], real_locs[i], showw = False)
        holders.append(holder)
    return np.array(holders)
    


###################################################################
# FUNCTION DEFINITIONS - dealing with intensities
###################################################################


# used to extract OG pixels and red pixels from the folder
def from_folder_getem(here):

    OG_path = here + "OG_GIF.gif"
    RED_path = here + "new_red.gif"

    OG = change3to1(extract_gif(OG_path))
    RED = change3to3(extract_gif(RED_path))

    return OG, RED

# takens in all gif pics and iso to make vector of intensity values
def intensity_extraction(OG, RED):

    # use iso to define lengths
    iso_path = here + "iso.png"
    iso = change4to1(image_to_pixels(iso_path))
    x, y = little_iso_range(iso)


    print("extracting intensity vector...")
    tic = time.perf_counter()
    vec = []
    for w in range(len(OG)):
        frame = OG[w]
        red = RED[w]
        
        sum = 0
        count = 0

        for i in range(x[0], x[1]+1, 1):
            for j in range(y[0], y[1]+1, 1):

                # print(count, w, i, j, red[i][j], frame[i][j])
                if np.array_equiv(red[i][j],[255, 255, 255]):
                    sum = sum + frame[i][j]
                    count += 1
                else:
                    pass
        
        vec.append(round(sum/count))
    
    toc = time.perf_counter()
    tictoc(tic, toc)
    print("intensities extracted.\n")
    return vec

# takes in intensity vectors and graphs with respect to 0.1 s time
def TIC_maker(TIC_vec, here):
    print("making TIC plot...")
    tic = time.perf_counter()
    x = []
    # first make the x vector:
    for i in range(len(TIC_vec)):
        x.append(float("%.1f" % (i * (0.1))))
    
    title = "Time Intensity Curve (TIC) of Slice - Case 1 CABG Project"
    
    plt.plot(x, TIC_vec, label='TIC')
    plt.xlabel("time of perfusion acquisition [s]")
    plt.ylabel("intensity of the pixels [grayscale]")
    plt.title(title)
    plt.legend()
    plt.xlim(0, 16)
    plt.ylim(0, 255)

    name = here + "TIC.png"
    plt.savefig(name)
    plt.show()

    # while plotting, let's save the data to a txt file too!
    name1 = "time of perfusion acquisition [s]"
    thing1 = x
    name2 = "intensity of the pixels [grayscale]"
    thing2 = TIC_vec
    name = here + "/plotresults.txt"
    textfile = open(name, "w")
    textfile.write(name1 + "\n")
    for element in thing1:
        textfile.write(str(element) + "\n")
    textfile.write("\n")
    textfile.write(name2 + "\n")
    for element in thing2:
        textfile.write(str(element) + "\n")
    textfile.close()

    toc = time.perf_counter()
    tictoc(tic, toc)
    print("plot created and saved.\n")
    
    return x

# the last step is to make the TICS! and save the data as a sheet too
def run_TIC_creator(here, continuation = "False",new_reds = [], pixel_grid = []):
    
    # we're running the full script so already saved
    if continuation == True:
        OG = pixel_grid
        RED = new_reds
    else:
        OG, RED = from_folder_getem(here)

    TIC_vec = intensity_extraction(OG, RED)
    TIC_maker(TIC_vec, here)

    return





###################################################################
# MAIN
###################################################################
aaa = time.perf_counter()

# outlines everywhere we should save stuff
# gif_path = "/Users/felicialiu/Desktop/summer2022/coding/gifs/AORTAslice20frames11.gif"
gif_path = "/Users/felicialiu/Desktop/summer2022/coding/gifs/AORTAslice90frames90.gif"
# gif_path = "/Users/felicialiu/Desktop/summer2022/coding/gifs/AORTAslice90frames2.gif"
parent_dir = "/Users/felicialiu/Desktop/summer2022/coding/movingbolus_test"

# first we make the directory to save everything to
here = make_dir(parent_dir, gif_path)
# here = "/Users/felicialiu/Desktop/summer2022/coding/movingbolus_test/AORTAslice90frames2NEWEST/"

# then we extract the gif
pixel_grid = change3to1(extract_gif(gif_path))
# pixel_grid = pixel_grid[:35]
# pixel_grid = time_clipper([10, 20, 30, 40, 50, 60, 70], pixel_grid)

# the save the gif to this new folder
speed = 80
save_the_gif(pixel_grid, "OG_GIF", speed, here)

# flattener needs the body to be in 1
flat = flattener(pixel_grid)
# printer(flat)
saver(flat, "flat", here)

# then we make iso and something to extract it
# this is after it's been cropped and masked
input("ready with iso?:  ")
amount_big = 80
cropped, rangee = iso_dealer(pixel_grid, here, amount_big)
cropped = change3to1(cropped)
save_the_gif(cropped, "cropped", speed, here)

# now we want to figure out the locations for each
thresholds, init_centre, ints, peak = the_decider(cropped, here)

# the fun part where locations and sizes are extracted
amount = 8
amount_in = 10
amount_up = 5

# locs_vec, sizes = whole_runner(test_pics, init_centre, thresholds, amount, here, peak)
test_pics = change1to3(cropped)
locs_vec, sizes = whole_runner2(test_pics, init_centre, thresholds, amount, here, peak, amount_in, amount_up)

# now we want to take those locations and remake them to the real one
real_locs = location_fixer(locs_vec, rangee)
pixel_grid = change1to3(pixel_grid)
new_reds = real_locs_shower(real_locs, pixel_grid)
save_the_gif(new_reds, "new_red", speed, here)
continuation = "True"

# last thing to do is make the TIC
run_TIC_creator(here, continuation)



bbb = time.perf_counter()
###################################################################
# CLEAN UP
###################################################################

# just to mark the end of the script

print("\n\nWHOLE RUNTIME:")
tictoc(aaa, bbb)
print("done")

print("\n\n#######################################################")