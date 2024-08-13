'''
GOAL:
- we're gonna properly get a bolus to fill in using polar coordindates maybe?
'''


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
# ITS FOR ARRAYS HERE
def change4to1(pixels):
    # print("converting [ , , , 255] to single...")
    tic = time.perf_counter()
    sheet = []
    for j in range(len(pixels)):
        line = []
        for k in range(len(pixels[j])):
            value = int(pixels[j][k][0])
            line.append(value)
        sheet.append(line)
    toc = time.perf_counter()
    # print("storage conversion complete.")
    # tictoc(tic, toc)
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
    string = ""
    for i in range(len(gif_path)-1, -1, -1):
        if gif_path[i] != "/":
            string = gif_path[i] + string
        else:
            break
    # new = gif_path.replace("/Users/felicialiu/Desktop/summer2022/coding/gifs/", "")
    new = string.replace(".gif", "")
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
# ENTERING PIXELS SHOULD BE THREE
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

# removes duplicate entries in list
def listremovedoubles(list):
    new = []
    for ele in list:
        if ele not in new:
            new.append(ele)
    return new












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
def make_dir_gif(parent_dir, gif_path):

    # start making folder with choose and end_range
    print("making directory...")
    directory = extract_gif_name(gif_path)
    path = os.path.join(parent_dir, directory) 
    os.mkdir(path) 
    here = parent_dir + "/" + directory + "/"
    print("directory '% s' created.\n" % directory)
    
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
# FUNCTION DEFINITIONS - for moving thresholds
###################################################################

# selects all the pixels around a small square
def small_square_around(center, size, pixels):
    locs = []
    for x in range(center[0]-round(size/2), center[0]+round(size/2), 1):
        for y in range(center[1]-round(size/2), center[1]+round(size/2), 1):
            locs.append([x, y])
    
    # the for each loc we find the intensity and report as sum
    sum = 0
    for loc in locs:
        sum = sum + pixels[loc[0]][loc[1]]
    intensity = round(sum/len(locs))
    thres = intensity-23
    return thres








###################################################################
# FUNCTION DEFINITIONS - running my polar thing
###################################################################

# turn degrees to radians
def rad(x):
    return float(x*math.pi/180)
def deg(x):
    return float(x*180/math.pi)

# turn rectangular and polar coordinates
def rect(pol):
    theta = rad(pol[0])
    r = pol[1]
    x = float("%.2f" % (r*math.cos(theta)))
    y = float("%.2f" % (r*math.sin(theta)))
    return [x, y]
def pol(rect):
    x = rect[0]
    y = rect[1]

    # deals with domains
    if x <= 0:
        mul = 180
    else:
        mul = 0

    if x == 0:
        if y > 0:
            theta = 90
        else:
            theta = 270
    else:
        theta = float("%.2f" % (mul + deg(math.atan(y/x))))

    # corrects for negative angles
    if theta < 0:
        theta = 360 + theta

    r = float("%.2f" % (math.sqrt(x**2 + y**2)))
    return [theta, r]
def rectwhole(pol):
    theta = rad(pol[0])
    r = pol[1]
    x = round(r*math.cos(theta))
    y = round((r*math.sin(theta)))
    return [x, y]

# measures point distances
def dist(point1, point2):
    x = point2[0] - point1[0]
    y = point2[1] - point1[1]
    return float("%.2f" % (math.sqrt(x**2 + y**2)))

# makes sure the pixel is on the grid
# or returns the next best thing
def grid_checker(x, y, pixels):
    point = [x, y]
    xx = x
    yy = y
    needed = False
    if point[0] > len(pixels)-1:
        xx = len(pixels)-1
        needed = True
    if point[0] < 0:
        xx = 0
        needed = True
    if point[1] > len(pixels[0])-1:
        yy = len(pixels[0])-1
        needed = True
    if point[1] < 0:
        yy = 0
        needed = True
    return xx, yy, needed

# so the polarfiller idea is to extend as long as possible in one direction
# maybe we do it twice to confirm and only use points in both 
# start_dir is the angle 
def polaredge(pixels, centre, thres, closeness, start_dir):
    # print("finding rough edge...")
    locations = []
    centres = [centre]
    
    # set up a vector of angles to look at
    angles = []
    for i in range(0, 360, closeness):
        angles.append(i)

    # change it so we start at the right angle
    for i in range(len(angles)):
        angles[i] = angles[i] + start_dir
        if angles[i] >= 360:
            angles[i] = angles[i] - 360

    for angle in angles:
        # we're gonna increase the distance one at a time until the brightness is bad
        dist = 0
        measure = 255
        while measure > thres: # while the brightness is good
            dist += 1
            rectt = rect([angle, dist])
            x = round(centre[0]-rectt[1])
            y = round(centre[1]+rectt[0])
            x, y, needed = grid_checker(x, y, pixels)
            if needed == True:
                break
            measure = pixels[x][y]
        locations.append([x, y])
        # selected_shower(pixels, locations, centres, typee = "single")
    return locations

# returns indices for longest chain of zeros front and back
def longest_chain(jumps):

    one = 0
    two = 0
    length = 0
    longest_length = 0
    lone = 0
    ltwo = 0
    for i in range(len(jumps)):
        if jumps[i] == 0:
            length += 1
            two = i
        if jumps[i] == 1:
            length = 0
            one = i+1
            two = i+1
        if length > longest_length:
            lone = one
            ltwo = two
            longest_length = length
    return lone, ltwo

# will choose the appropriate cutoff
def cutoffdecider(distss):

    dists = []
    for dist in distss:
        dists.append(dist)
    dists.sort()
    # plainplot(dists)

    # the first jump of more than 5 pixels is the cutoff?
    jumps = []
    for i in range(len(dists)-1):
        value = dists[i+1]-dists[i]
        if value > 3:
            jumps.append(1)
        else:
            jumps.append(0)
    # plainplot(jumps)

    # keep track of the longest set of small jumps
    one, two = longest_chain(jumps)
        
    front1 = dists[one]
    back1 = dists[one+1]
    front2 = dists[two]
    back2 = dists[two+1]

    small = round((back1+front1)/2)
    big = round((back2+front2)/2)

    return small, big

# let's the user control which points are outside the contour
def remove_bad(centre, locations):
    # print("removing bad edge points...")
    dists = []
    for loc in locations:
        dists.append(dist(centre, loc))

    # # already cutoff will check if we need input or if it already has one
    # if already_cutoff == 0:
    #     cutoff = input("choose cutoff:  ")
    # else:
    #     cutoff = already_cutoff

    front, back = cutoffdecider(dists)
    
    # now we cutoff the locations
    new_locations = []
    for i in range(len(dists)):
        if dists[i] < float(back):
            if dists[i] > float(front):
                new_locations.append(locations[i])
    return new_locations

# will choose the appropriate cutoff
def possible_cutoffs(distss):

    dists = []
    for dist in distss:
        dists.append(dist)
    dists.sort()
    plainplot(dists)

    # the first jump of more than 5 pixels is the cutoff?
    jumps = []
    for i in range(len(dists)-1):
        jumps.append(dists[i+1]-dists[i])
    # plainplot(jumps)

    count = -1
    poss = []
    for i in range(len(jumps)):
        if jumps[i] > 3:
            count = i
            poss.append((dists[count]+dists[count+1])/2)
            
    if count == -1: # means no significant jumps, all points included
        poss.append(dists[-1] + 1)
        
    return poss

# might need a different method to remove the bad ones
def remove_bad2(centre, locations, already, min_dist, max_dist):
    dists = []
    for loc in locations:
        dists.append(dist(centre, loc))
    
    if already == False: # there are no previous distances
        return remove_bad(centre, locations)
    
    # get a list of possible values
    # poss = possible_cutoffs(dists)
    # print(poss)

    # otherwise we'll assume the points need to be within the other one's range
    new_locations = []
    for i in range(len(dists)):
        if dists[i] < float(max_dist)+2:
            if dists[i] > float(min_dist)-2:
                new_locations.append(locations[i])

    # in these new points we can see if there's any weird jumps
    dists = []
    for loc in new_locations:
        dists.append(dist(centre, loc))
    
    # now we can check for big jumps
    newer_locations = []
    for i in range(1, len(dists)-1, 1):
        past = dists[i] - dists[i-1]
        next = dists[i] - dists[i+1]
        avg = (past+next)/2
        if avg < 3: # no significan't jumping
            newer_locations.append(new_locations[i])

    return newer_locations
    
# look at the 8 locations around the point, if exceeding threshold
# it's valuable to keep, to add to the queue, else we discard
def find_all_around(point):
    row = point[0]
    col = point[1]

    # open set holds all the points around
    open_set = [point]
    open_set.append([row, col-1]) #LL
    open_set.append([row-1, col-1]) #TL
    open_set.append([row-1, col]) #TT
    open_set.append([row-1, col+1]) #TR
    open_set.append([row, col+1]) #RR
    open_set.append([row+1, col+1]) #BR
    open_set.append([row+1, col]) #BB
    open_set.append([row+1, col-1]) #BL

    random.shuffle(open_set)
    return open_set

# returns true or false depending on if there's a solid path
# between all the points in a vector
def travellable(points):
    for i in range(len(points)-1):
        point = points[i]
        next = points[i+1]
        set = find_all_around(point)
        if next not in set:
            return False
    return True
        
# figures out a full path from one point to the other
# CIRCULAR
def interprolate(point1, point2, cen):

    # first form the right vectors
    vec1 = pol([point1[1]-cen[1], -(point1[0]-cen[0])])
    vec2 = pol([point2[1]-cen[1], -(point2[0]-cen[0])])

    if vec1[0] > vec2[0]:
        vec2[0] = vec2[0] + 360

    # surely the distance can't be more than 1.5 times shortest between
    distt = dist(point1, point2)
    
    # making the locations
    mul = 2
    angles = np.linspace(vec1[0], vec2[0], round(distt*mul))
    rs = np.linspace(vec1[1], vec2[1], round(distt*mul))
    vec = []
    for i in range(len(angles)):
        x = angles[i]
        y = rs[i]
        rectt = rectwhole([angles[i], rs[i]])
        x = round(cen[0]-rectt[1])
        y = round(cen[1]+rectt[0])
        vec.append([x, y])
    return vec

# figures out a full path from one point to the other
# STRAIGHT LINE
def interprolate2(point1, point2, cen):

    # making the locations
    distt = dist(point1, point2)
    mul = 2
    xs = np.linspace(point1[0], point2[0], round(distt*mul))
    ys = np.linspace(point1[1], point2[1], round(distt*mul))
    vec = []
    for i in range(len(xs)):
        x = round(xs[i])
        y = round(ys[i])
        vec.append([x, y])
    return vec

# figures out a full path from one point to the other
# CIRCLE + LINE BETWEEN 2/3 to circle
# runs interprolate 1 and 2 and averages all the points
def interprolate3(point1, point2, cen):
    ones = interprolate(point1, point2, cen)
    twos = interprolate2(point1, point2, cen)

    threes = []
    if len(ones) == len(twos):
        for i in range(len(ones)):
            one = ones[i]
            two = twos[i]
            three = [round((2*one[0]+two[0])/3), round((2*one[1]+two[1])/3)]
            threes.append(three)
    else:
        threes = twos

    return threes

# idea is to interprolate between dots and form a contour to be filled in
# we might not even take into account the intensities?
def contouring(locations, centre):
    # print("creating contour from guideline points...")

    locs = []
    for i in range(len(locations)-1):
        point1 = locations[i]
        point2 = locations[i+1]
        addins = interprolate(point1, point2, centre)
        # put those into the whole
        for i in addins:
            locs.append(i)
    # last we need to seal the circle
    point1 = locations[-1]
    point2 = locations[0]
    addins = interprolate(point1, point2, centre)
    for i in addins:
        locs.append(i)
    # some clean up stuff
    locs = listremovedoubles(locs)
    if travellable(locs) == False:
        print("*** ATTENTION - NOT FULL CONTOUR ***")

    return locs
def contouring2(locations, centre):
    # print("creating contour from guideline points...")

    locs = []
    for i in range(len(locations)-1):
        point1 = locations[i]
        point2 = locations[i+1]
        addins = interprolate2(point1, point2, centre)
        # put those into the whole
        for i in addins:
            locs.append(i)
    # last we need to seal the circle
    point1 = locations[-1]
    point2 = locations[0]
    addins = interprolate2(point1, point2, centre)
    for i in addins:
        locs.append(i)
    # some clean up stuff
    locs = listremovedoubles(locs)
    if travellable(locs) == False:
        print("*** ATTENTION - NOT FULL CONTOUR ***")

    return locs
def contouring3(locations, centre):
    # print("creating contour from guideline points...")

    locs = []
    for i in range(len(locations)-1):
        point1 = locations[i]
        point2 = locations[i+1]
        addins = interprolate3(point1, point2, centre)
        # put those into the whole
        for i in addins:
            locs.append(i)
    # last we need to seal the circle
    point1 = locations[-1]
    point2 = locations[0]
    addins = interprolate3(point1, point2, centre)
    for i in addins:
        locs.append(i)
    # some clean up stuff
    locs = listremovedoubles(locs)
    if travellable(locs) == False:
        print("*** ATTENTION - NOT FULL CONTOUR ***")

    return locs

# fills in an entire closed contour
def filling_it_in(locations):

    # if travellable(locations) == False:
    #     print("*** ATTENTION - NOT FULL CONTOUR ***")
    #     return locations

    # locations currently go in a circle so we should sort them
    # top to bottom, left to right?
    locations.sort()

    # making a fake one
    locs = []
    for loc in locations:
        locs.append(loc)

    leaving = []
    for i in range(locations[0][0], locations[-1][0]+1, 1):

        # for each index, we wanna figure out where to split locs
        count = 0
        for j in range(len(locs)):
            if locs[j][0] != i:
                break
            count += 1
        use = locs[:count]
        next = locs[count:]

        # check if we're on a valid number
        if use == []:
            continue

        # on use we then want to fill it in
        start = use[0][1]
        end = use[-1][1]
        addees = []
        for num in range(start, end+1, 1):
            addees.append([i, num])
        
        # these addees need to go in the master list
        for thing in addees:
            leaving.append(thing)

        # setting it up for the next round
        locs = next
    return leaving

# takes on pixel slice and for every point on it not in locations
# the pixel is dimmed by some amount
def grid_masker2(slice, locations, amount):
    for i in range(len(slice)):
        for j in range(len(slice[i])):
            locloc = [i, j]
            if locloc not in locations: # meaning it's outside, we want to mask it
                value = max(round(slice[i][j] - amount), 0)
                slice[i][j] = value
            else: # it's in locations so it can get a little brighter
                value = min(round(slice[i][j] + amount), 255)
                slice[i][j] = value
    return slice

# looks at the time pics and applys a darkening mask to all the pixels 
# outside of the locations by the given amount for the rest of time
def the_following_get_darker2(pixel_grid, behind_this, locations, amount, chop_factor):
    print("masking pictures", behind_this+1, "to", len(pixel_grid)-1, "by", amount, "intensity...")
    tic = time.perf_counter()

    for w in range(behind_this+1, min(behind_this+chop_factor+1, len(pixel_grid)), 1):
        print("masking", w)
        pixel_grid[w] = grid_masker2(pixel_grid[w], locations, amount)

    toc = time.perf_counter()
    print("masking done.")
    tictoc(tic, toc)
    return pixel_grid

# looks for boundaries for next iteration
def find_dist(locations, centre):
    dists = []
    for loc in locations:
        dists.append(dist(centre, loc))
    return np.min(dists), np.max(dists)








###################################################################
# FUNCTION DEFINITIONS - intensity stuff
###################################################################

# finds intensity for one plane using locations
def loc_pixels_intensity_extraction(locations, pixels):
    sum = 0
    count = 0
    for loc in locations:
        down = loc[0]
        side = loc[1]
        sum = sum + pixels[down][side]
        count += 1
    return round(sum/count)

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
    # plt.show()
    plt.clf()

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





###################################################################
# FUNCTION DEFINITIONS - MAIN
###################################################################

def the_whole_polarthing(gif_path, parent_dir, speed):

    # first we make the directory to save everything to
    here = make_dir_gif(parent_dir, gif_path)
    # here = "/Users/felicialiu/Desktop/summer2022/coding/polarfiller/AORTA34slice/"

    # then we extract the gif
    pixel_grid = change3to1(extract_gif(gif_path))
    # pixel_grid = time_clipper([0,1,2,3,4,5,6,7,8,9,10], pixel_grid)

    # the save the gif to this new folder
    save_the_gif(pixel_grid, "OG_GIF", speed, here)

    # flattener needs the body to be in 1
    flat = flattener(pixel_grid)
    printer(flat)
    saver(flat, "flat", here)

    # THESE ARE USER INPUTS
    down = input("down:  ")
    side = input("side  ")
    centre = [int(down), int(side)]
    # centre = [100, 80] # THIS IS USER INPUT

    # these you don't need to touch
    centres = [centre]
    size = 10 # for the square to find threshold
    closeness = 5 # degrees apart
    start_dir = 0
    already = False
    min_dist = 0 
    max_dist = 0
    
    reds = []
    cont = []
    dots = []
    starter = []
    TIC_vec = []

    for i in range(len(pixel_grid)):
        print("running intensity for", i)
        pixels = pixel_grid[i]

        # first step only extracts a bare minimum edge including bad points
        thres = small_square_around(centre, size, pixels)
        locations = polaredge(pixels, centre, thres, closeness, start_dir)
        oooh = selected_shower(pixels, locations, centres, typee = "single", showw = False)
        starter.append(oooh)

        # now we want a function to examine and smartly remove bad points
        locations = remove_bad2(centre, locations, already, min_dist, max_dist)
        already = True
        wow = selected_shower(pixels, locations, centres, typee = "single", showw = False)
        dots.append(wow)

        # now we can take the set of dots and form a contour?
        locations = contouring3(locations, centre)
        locations = contouring2(locations, centre) # will prevent weird breaks
        heap = selected_shower(pixels, locations, centres, typee = "single", showw = False)
        cont.append(heap)

        # we'll use the outside contour to find the min and max dist for next time
        min_dist, max_dist = find_dist(locations, centre)

        # now we can fill in the contour
        locations = filling_it_in(locations)
        holder = selected_shower(pixels, locations, centres, typee = "single", showw = False)
        reds.append(holder)

        # for the locations we want to extract the intensity and add to vector
        intensity = loc_pixels_intensity_extraction(locations, pixels)
        TIC_vec.append(intensity)

        # masking pixels wasn't helping
        # pixel_grid = the_following_get_darker2(pixel_grid, i, locations, amount, 3)

    reds = np.array(reds)
    save_the_gif(reds, "RED", speed, here)
    cont = np.array(cont)
    save_the_gif(cont, "CONTOUR", speed, here)
    dots = np.array(dots)
    save_the_gif(dots, "GOOD_POINTS", speed, here)
    starter = np.array(starter)
    save_the_gif(starter, "ALL_POINTS", speed, here)

    # and we need to make the TIC_vec!
    TIC_maker(TIC_vec, here)

    return

def run_tics_for_all_gifs(parent_dir, speed):
    gif_folder_path = parent_dir + "/gif/"
    names = os.listdir(gif_folder_path)
    
    for name in names:
        gif_path = gif_folder_path + name
        print("*******\nNOW WORKING ON", name)
        the_whole_polarthing(gif_path, parent_dir, speed)



###################################################################
# MAIN
###################################################################
aaa = time.perf_counter()

# we'll use this once the first tests are all working!

# outlines everywhere we should save stuff
gif_path = "/Users/felicialiu/Desktop/summer2022/coding/runthrough2/gif/AORTA2slice.gif"
parent_dir = "/Users/felicialiu/Desktop/summer2022/coding/polarfiller"
# parent_dir = "/Users/felicialiu/Desktop/summer2022/coding/runthrough2"
speed = 80
the_whole_polarthing(gif_path, parent_dir, speed)



# run_tics_for_all_gifs(parent_dir, speed)








# # currently we're just gonna work with single frames
# # they are all single intensity values 
# pixel_grid = change3to1(extract_gif(gif_path))
# pixel_grid = time_clipper([2, 3, 6], pixel_grid)
# pic1 = pixel_grid[0]
# pic2 = pixel_grid[1]

# # first step only extracts a bare minimum edge including bad points
# pixels = pic1
# centre = [100, 130]
# centres = [centre]
# size = 10
# thres = small_square_around(centre, size, pixels)
# closeness = 1 # degrees apart
# start_dir = 45

# locations = polaredge(pixels, centre, thres, closeness, start_dir)
# selected_shower(pixels, locations, centres, typee = "single")

# # now we want a function to examine and smartly remove bad points
# locations, cutoff = remove_bad2(centre, locations)
# selected_shower(pixels, locations, centres, typee = "single")

# # now we can take the set of dots and form a contour?
# locations = contouring2(locations, centre)
# selected_shower(pixels, locations, centres, typee = "single")

# # now we can fill in the contour
# locations = filling_it_in(locations)
# selected_shower(pixels, locations, centres, typee = "single")







# # pictures to unload
# path1 = "/Users/felicialiu/Desktop/summer2022/coding/polarfiller/testcases/contour.png"
# path2 = "/Users/felicialiu/Desktop/summer2022/coding/polarfiller/testcases/notouchingeasy.png"
# path3 = "/Users/felicialiu/Desktop/summer2022/coding/polarfiller/testcases/notouchinghard.png"
# path4 = "/Users/felicialiu/Desktop/summer2022/coding/polarfiller/testcases/notouching.png"
# contour = change4to1(image_to_pixels(path1))
# first = change4to1(image_to_pixels(path4))
# easy = change4to1(image_to_pixels(path2))
# hard = change4to1(image_to_pixels(path3))

# # first step only extracts a bare minimum edge including bad points
# pixels = hard
# centre = [40, 40]
# centres = [centre]
# thres = 250
# closeness = 1 # degrees apart
# start_dir = 45

# locations = polaredge(pixels, centre, thres, closeness, start_dir)
# selected_shower(pixels, locations, centres, typee = "single")

# # now we want a function to examine and smartly remove bad points
# locations, cutoff = remove_bad(centre, locations)
# selected_shower(pixels, locations, centres, typee = "single")

# # now we can take the set of dots and form a contour?
# locations = contouring2(locations, centre)
# print(locations)
# selected_shower(pixels, locations, centres, typee = "single")

# # now we can fill in the contour
# locations = filling_it_in(locations)
# selected_shower(pixels, locations, centres, typee = "single")


bbb = time.perf_counter()
###################################################################
# CLEAN UP
###################################################################

# just to mark the end of the script

print("\n\nWHOLE RUNTIME:")
tictoc(aaa, bbb)
print("done")

print("\n\n#######################################################")