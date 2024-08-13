'''
the point of this is that the SCALING needs to be right...
using the meta data, which i also need to check if it's the 
same for all dicoms of the same paitent over time....
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

# choosing function just makes a list of skip counting
def choosing(start, end, skip):
    list = []
    for i in range(start, end, skip):
        list.append(i)
    return list

# finds average value of a list
def avgoflist(L):
    sum = 0
    count = 0
    for i in range(len(L)):
        sum = sum + L[i]
        count += 1
    return round(sum/count)





###################################################################
# FUNCTION DEFINITIONS - 
###################################################################




###################################################################
# FUNCTION DEFINITIONS - everything needed to extract gifs!
###################################################################

# for each gif, create it's own directory to put all its stuff into
# and first, stick the gif in it!
def make_dir(parent_dir, name):
    print("making directory", name, "...")
    directory = name
    path = os.path.join(parent_dir, directory) 
    os.mkdir(path) 
    here = parent_dir + "/" + directory + "/"
    print(here)
    print("directory '% s' created.\n" % directory)
    return here

# takes in a folder with all the 3D dicom images and puts all into array
def extract_4DCTpixels(folder, check, woow):
    print("generating 4D CT pixel grid...")
    tic = time.perf_counter()

    if check == False:
        end_range = woow
    else:
        end_range = len(os.listdir(folder))
    holder = []

    hundreds = 0
    tens = 0
    ones = 1

    # for each DICOM in the whole folder, we're going to extract one slice
    for i in range(1, end_range+1):

        # just prepares the name of the file
        num_in = str(hundreds) + str(tens) + str(ones)
        name = folder + "/IM-0001-0" + num_in + "-0001.dcm"
        ones += 1
        if ones == 10:
            tens += 1
            ones = 0
            if tens == 10:
                hundreds += 1
                tens = 0
        
        # now we fix the colour and append the new grid to a premade list
        # print(name)
        x = dcmread(name) 
        pixels = x.pixel_array
        holder.append(pixels)
        
    print("turning to numpy array...")
    # at the end, make holder an array
    holder = np.array(holder) 

    toc = time.perf_counter()
    print("done 4D CT extraction.")
    tictoc(tic, toc)
    return holder

# feed it the ugly .pth text and it generates a list for all id of the chunks
def segment_words(ugly):

    holder = ugly 
    extension = []

    while holder.find("<path_point id=") != -1:

        f = holder.index("<path_point id=")
        g = holder.index("</path_point>")
        # print("segmenting indices:", f, "to", g)

        new = holder[f : g+13]
        extension.append(new)
        holder = holder[g+13:]

    return extension

# s is a string and ch is the character in the str you want to find
# will report list of all indices
def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

# take a str and makes a float rounding to 4 decimals
def make_a_float(strr):
    # check negative:
    if strr[0] == "-":
        neg = True
        strr = strr[1:]
    else:
        neg = False
    
    strr = float(strr)
    strr = round(strr, 4)

    if neg == True:
        strr = -1 * strr
    
    return strr

# takes in one string starting with "path_point" ending with "/path_point"
# creates 4 outputs: idd, pos, tan, rot
def extracting_values(strr):

    list = find(strr, '"')
    
    if len(list) != 20:
        print("issue! abort.")
        return 0
    
    start = 0
    end = 1
    which = "idd"
    count = 0
    pos = []
    tan = []
    rot = []

    for i in range(10):
        
        if list[start] == list[end]:
            value = strr[list[start]+1]
        else:
            value = strr[list[start]+1:list[end]]
        
        if which == "idd":
            value = int(value)
            idd = value
            which = "pos"

        elif which == "pos":
            value = make_a_float(value)
            pos.append(value)
            count += 1
            if count == 3:
                count = 0
                which = "tan"
        
        elif which == "tan":
            value = make_a_float(value)
            tan.append(value)
            count += 1
            if count == 3:
                count = 0
                which = "rot"
        
        elif which == "rot":
            value = make_a_float(value)
            rot.append(value)
        
        start += 2
        end += 2
    return idd, [pos, tan, rot]

# for every path points, start a dictionary entry to hold:
    # id0 id1 id2 as the key   
    # then the values list will have 3 tupples, (x,y,z)
    # for position, tangent and rotation
# given a path to the .pth file, make a dictionary of everything!
def init_path_file(path):
    print("creating full centreline points dictionary...")
    tic = time.perf_counter()

    file = open(path)
    ugly = file.read()
    new = segment_words(ugly)

    dict = {}

    for i in range(len(new)):
        s, wow = extracting_values(new[i])
        name = "point" + str(s)
        dict[name] = wow
    
    print("done making dictionary.")
    toc = time.perf_counter()
    tictoc(tic, toc)
    return dict

# based on how the loc(mm) map to the ct dimensions
# linear recorrection of the locations to get matching!
# 1s are all [loc(mm) min and max]
# 2s are all [ct dim min and max] aka 512x512x480
def mapper(point, sag1, sag2, cor1, cor2, ax1, ax2):
   # 3.659, -1.3443, 225.1
   # first point is sagital control
   # second point is coronal control
   # third point is axial control
   sag = point[0]
   cor = point[1]
   ax = point[2]
   newsag = round(((sag-sag1[0])/(sag1[1]-sag1[0]))*(sag2[1]-sag2[0])+sag2[0])
   newcor = round(((cor-cor1[0])/(cor1[1]-cor1[0]))*(cor2[1]-cor2[0])+cor2[0])
   newax = round(((ax-ax1[0])/(ax1[1]-ax1[0]))*(ax2[1]-ax2[0])+ax2[0])

   return [newsag, newcor, newax]

# inner function to apply mapper to each point and define boundries
def mini_map(linepts, pixel_grid, values):
    # all the ones i'm getting from simvascular's image navigator
    sag1 = values[0] #[-9.19, 16.66]
    sag2 = [0, len(pixel_grid[0])-1]
    cor1 = values[1] #[-10.76, 15.10]
    cor2 = [0, len(pixel_grid[0][0])-1]
    ax1 = values[2] #[205.00, 220.90]
    ax2 = [0, int(len(pixel_grid)/3)-1]

    newpts = []
    for i in linepts:
        point = i
        newpts.append(mapper(point, sag1, sag2, cor1, cor2, ax1, ax2))
    
    nnewpts = []
    for newpt in newpts:
        nnewpts.append([newpt[0], newpt[1], newpt[2]*3+1])

    return nnewpts

# uses mini_map to run it for the entire dictionary
def position_remapper(dict, pixel_grid, values):
    linepts = []
    for key in dict:
        linepts.append(dict[key][0])

    newpts = mini_map(linepts, pixel_grid, values)
    # remapped_sanitycheck(pixel_grid, newpts)

    return newpts
  
# to fix the dimensions sorta x 3
def extend_3Dbody(pixel_grid):
    new = []

    # METHOD #1 -> simple elogation
    for i in pixel_grid:
        new.append(i)
        new.append(i)
        new.append(i)

    # ## METHOD #2 -> interprolation
    # for i in range(len(pixel_grid)-1):
    #     print("extending", i)
    #     now = pixel_grid[i]
    #     next = pixel_grid[i+1]

    #     shield1 = []
    #     shield2 = []
    #     for x in range(len(pixel_grid[i])):
    #         row1 = []
    #         row2 = []
    #         for y in range(len(pixel_grid[i][x])):
    #             val1 = round((next[x][y] - now[x][y])*(1/3) + now[x][y])
    #             val2 = round((next[x][y] - now[x][y])*(2/3) + now[x][y])
    #             row1.append(val1)
    #             row2.append(val2)
    #         shield1.append(row1)
    #         shield2.append(row2)
        
    #     new.append(now)
    #     new.append(np.array(shield1))
    #     new.append(np.array(shield2))
    # new.append(pixel_grid[len(pixel_grid)-1])
    # new.append(pixel_grid[len(pixel_grid)-1])
    # new.append(pixel_grid[len(pixel_grid)-1])
    # print(len(new))

    new = np.array(new)
    return new

# can choose anywhere from 1 to 99
def make_normal_simpler(newpts, loc):

    # sanity check to make sure location is valid
    if loc < 1 or loc > len(newpts)-2:
        print(loc, "is not a valid location. must be between 1 and", len(newpts)-2)
        return [0, 0, 0], [0, 0, 0], False
    
    lpos = newpts[loc-1]
    point = newpts[loc]
    mpos = newpts[loc+1]
    normal = [mpos[0] - lpos[0], mpos[1] - lpos[1] , mpos[2] - lpos[2]]
    return point, normal, True
    
# makes it's magnitude = 1
def normalise_vec(vec):
    mag = math.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)

    if mag == 0:
        return vec

    mult = 1/mag
    vec = [element * mult for element in vec]
    return vec

# given upwards facing normal vector, find the up vector
def upper(normal):
    x = -1 * normal[0]
    y = -1 * normal[1]
    z = (normal[0] **2 + normal[1] **2) / normal[2]
    return normalise_vec([x, y, z])
    
# manually does cross product
def sider(normal, up):
    # we're going to cross product them sorta manually
    '''
    a       d       bf - ce
    b   x   e   =   cd - af
    c       f       ae - bd
    '''
    a = normal[0]
    b = normal[1]
    c = normal[2]
    d = up[0]
    e = up[1]
    f = up[2]

    x = b*f - c*e
    y = c*d - a*f
    z = a*e - b*d
    return normalise_vec([x,y,z])

# function will move us around the plane using our up and left vectors
# -up is down and -side is right
def m(start, num1, up, num2, side):
    # first do all our side moves then the up/down moves
    x = start[0] + num2*side[0] + num1*up[0]
    y = start[1] + num2*side[1] + num1*up[1]
    z = start[2] + num2*side[2] + num1*up[2]
    return [x, y, z]

# shows the frame on the vessel
def plot_graphwithvessel(locs, newpts):

    # see what it looks like the vessel
    x = []
    y = []
    z = []
    for i in newpts:
        x.append(i[0])
        y.append(i[1])
        z.append(i[2])

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    point1 = locs[0][0]
    point2 = locs[0][len(locs)-1]
    point3 = locs[len(locs)-1][len(locs)-1]
    point4 = locs[len(locs)-1][0]

    xx = [point1[0], point2[0], point3[0], point4[0], point1[0]]
    yy = [point1[1], point2[1], point3[1], point4[1], point1[1]]
    zz = [point1[2], point2[2], point3[2], point4[2], point1[2]]

    ax.plot3D(x, y, z, 'maroon')
    ax.plot3D(xx, yy, zz, 'green')
    ax.set_xlim3d(0, 512)
    ax.set_ylim3d(0, 512)
    ax.set_zlim3d(0, 512)
    ax.set_xlabel('sagital')
    ax.set_ylabel('coronal')
    ax.set_zlabel('axial')
    ax.set_title("line plot with normal plane")
    plt.show()

    return

# figures out the locations of each element on the noramal plane
def bring_back_locs(point, normal, size):

    normal = normalise_vec(normal)

    # with the size, first make two blank arrarys 
    # one to be the intensity values we'll fill in
    # one to be the 512x512 locations thing to shape to
    locs = np.array([[[0,0,0]]*size]*size)

    # then we'll fill the locs with correct locations based 
    # on the point and plane and where it is based one the CT's dim

    # get the up/down and side pointing vectors
    # first make sure normal vector is pointing upwards
    if normal[2] < 0:
        normal = [element * -1 for element in normal]
    # the other case is if the normal is only radially outwards
    # then the normal points in only the z direction
    if normal[2] == 0: 
        up = [0, 0, 1]
    # otherwise.... we're good to go!
    if normal[2] > 0:
        up = upper(normal)
    # the normal points straight up... which also isn't good
    if normal[0] == 0 and normal[1] == 0:
        up = [1, 0, 0]
        
    side = sider(normal, up)

    # now based on the size we'll fill in the whole locs array with locs!

    # # METHOD #1 -> go top left then move right then down all the way
    # # first we need a start location so point is in the middle
    # start = m(point, size/2, up, size/2, side) # <- this is the upper left

    # # for every row basically
    # for i in range(len(locs)):
    #     for j in range(len(locs[i])):
    #         value = m(start, 0, up, -1, side)
    #         locs[i][j] = value
    #         start = value
    #     # filling in a row, gotta go back all the way and move down one
    #     hold = m(start, -1, up, len(locs[i]), side)
    #     start = hold

    # # METHOD #2 -> assign point as 50 50, then fill in rest of it's row, then for each of those
    # fill their tops and bottoms

    # first we assign the centre to be this centre
    i = round(size/2)
    j = round(size/2)
    locs[i][j] = point

    # then we scroll through all js so back and forth for i = round(size/2)
    # this is moving to the left of the middle row
    start = point
    for j in range(round(size/2)-1, -1, -1):
        value = m(start, 0, up, 1, side)
        locs[i][j] = value
        start = value

    # this is moving to the right of the middle row
    start = point
    for j in range(round(size/2)+1, size, 1):
        value = m(start, 0, up, -1, side)
        locs[i][j] = value
        start = value

    # now we have a long middle row done, each will be the start of doing their column
    for j in range(len(locs[round(size/2)])):

        # the starting point will depend on the coloumn
        start = locs[round(size/2)][j]

        # this is moving to the up of the start point
        for i in range(round(size/2)-1, -1, -1):
            value = m(start, 1, up, 0, side)
            locs[i][j] = value
            start = value
        
        # this is moving to the right of the middle row
        start = locs[round(size/2)][j]
        for i in range(round(size/2)+1, size, 1):
            value = m(start, -1, up, 0, side)
            locs[i][j] = value
            start = value

    # plot_vectors_plane(normal, up, side, point, locs)
    return locs

# redefines pixels to be within 0 to 255
def my_colourmap(pixels):

    # low = np.amin(pixels)
    # max = np.amax(pixels)
    low = -300
    thres = 0
    high = 1000

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

# does interprolation based on it's location and the 
# whole pixelgrid to determine intensity
def localintensity(loc, pixel_grid):

    # first is the loc is off the 512x512x160*3 grid
    if loc[2] < 0 or loc[2] >= len(pixel_grid):
        return -2048
    if loc[0] < 0 or loc[0] >= len(pixel_grid[0]):
        return -2048
    if loc[1] < 0 or loc[1] >= len(pixel_grid[0][0]):
        return -2048
    
    # x is sagital y is coronal z is axial 
    return pixel_grid[loc[2]][loc[1]][loc[0]]
    # otherwise it must be on the grid, so we can guess it's intensity value!
    # say it's like [435, 218, 35] the point, what are it's neighbours intensities
    xs = math.floor(loc[0])
    xm = math.ceil(loc[0])
    ys = math.floor(loc[1])
    ym = math.ceil(loc[1])
    zs = math.floor(loc[2])
    zm = math.ceil(loc[2])

    # each point has 8 neighbours which we can get intensities for
    '''
    top : b     c     bot : f     g

          a     d           e     h
    '''
    a = pixel_grid[xm][ys][zm]
    b = pixel_grid[xs][ys][zm]
    c = pixel_grid[xs][ym][zm]
    d = pixel_grid[xm][ym][zm]
    e = pixel_grid[xm][ys][zs]
    f = pixel_grid[xs][ys][zs]
    g = pixel_grid[xs][ym][zs]
    h = pixel_grid[xm][ym][zs]

    # this interprolation is gonna check distances and do weighted average
    # favouring proximity
    xss = (loc[0]-xs)/(xm-xs)
    xmm = (xm-loc[0])/(xm-xs)
    yss = (loc[1]-ys)/(ym-ys)
    ymm = (ym-loc[1])/(ym-ys)
    zss = (loc[2]-zs)/(zm-zs)
    zmm = (zm-loc[2])/(zm-zs)
    print(loc)
    print(xs, xm, ys, ym, zs, zm)
    print(xss, xmm, yss, ymm, zss, zmm)
    totxs = (b+c+f+g)/4
    totxm = (a+d+e+h)/4
    totys = (a+b+e+f)/4
    totym = (c+d+g+h)/4
    totzs = (e+f+g+h)/4
    totzm = (a+b+c+d)/4

    intx = totxs*xss + totxm*xmm
    inty = totys*yss + totym*ymm
    intz = totzs*zss + totzm*zmm

    return (intx+inty+intz)/3

# overlay the normal plane and voxels to extract plane of right intensities
# inputs, elogated pixelgrid, the centreline point, its normal vector, 
# and size of the output image array -> to get the image array (square)
def voxel_plane_extraction_withlocs(pixel_grid, point, normal, size, locs):
    
    intensities = np.array([[0]*size]*size)
    
    # then for each location already set, we'll create the new grid of pixel values
    for i in range(len(locs)):
        for j in range(len(locs[i])):
            intensities[i][j] = localintensity(locs[i][j], pixel_grid)

    return intensities

# the main function that does a gif of one choose plane for given time
def a_plane_over_time(pixelgrid4D, TEST, choose, size, values):

    # also prep the pathline into a dictionary
    dict = init_path_file(TEST)

    # now we define what the vessel looks like at time zero
    newpts = position_remapper(dict, extend_3Dbody(pixelgrid4D[0]), values)
    
    # we can define the plane of interests points by getting the normal
    point, normal, worked = make_normal_simpler(newpts, choose)
    if worked == False:
        return [], False# the chosen index was out of range
    locs = bring_back_locs(point, normal, size)
    # plot_graphwithvessel(locs, newpts)

    # x = input("if bad locs look, [X] for out:  ")
    # if x == "X":
    #     print("please try new choose location.")
    #     return [], False
    
    # now for each body, we're going to extract those planes
    planetime_gif = []
    for i in range(len(pixelgrid4D)):
        print("extracting slide from pixel_grid",i+1)
        pixel_grid = extend_3Dbody(pixelgrid4D[i])
        intgrid = my_colourmap(voxel_plane_extraction_withlocs(pixel_grid, point, normal, size, locs))
        planetime_gif.append(intgrid)
        # printer2(intgrid)

    return planetime_gif, True

# collects to correct metadata values and puts them in that nice [sag, cor, ax]
def getallmetadata(folder, parent_dir, pixel_grid):

    name = folder + "/IM-0001-0001-0001.dcm"
    
    x = dcmread(name)
    metaname = parent_dir + "/metadata.txt"
    print(metaname)
    textfile = open(metaname, "w")
    textfile.write(str(x))
    textfile.close()

    file = open(metaname)
    ugly = file.readlines()
    
    for line in ugly:
        if "(0020, 0032) Image Position (Patient)" in line:
            fixfirst = line
            break
    
    for line in ugly:
        if "(0018, 9322) Reconstruction Pixel Spacing" in line:
            fixspace = line
            break

    # now we have the two write strings, gotta change them into vectors
    first = []
    ind1 = fixfirst.index("[")
    fixfirst = fixfirst[ind1:]
    ind1 = fixfirst.index(",")
    first.append(float(fixfirst[1:ind1]))
    fixfirst = fixfirst[ind1+2:]
    ind1 = fixfirst.index(",")
    first.append(float(fixfirst[0:ind1]))
    fixfirst = fixfirst[ind1+2:]
    ind1 = fixfirst.index("]")
    first.append(float(fixfirst[0:ind1]))
    space = []
    ind1 = fixspace.index("[")
    fixspace = fixspace[ind1:]
    ind1 = fixspace.index(",")
    space.append(float(fixspace[1:ind1]))
    fixspace = fixspace[ind1+2:]
    ind1 = fixspace.index("]")
    space.append(float(fixspace[0:ind1]))

    # and lastly we convert to use with the non-stretched grid
    a = len(pixel_grid[0])-1
    b = len(pixel_grid[0][0])-1
    c = len(pixel_grid)-1

    sag1 = [float("%.4f" % (first[0]*0.1)), float("%.4f" % (0.1*(first[0] + (a)*space[0])))]
    cor1 = [float("%.4f" % (first[1]*0.1)), float("%.4f" % (0.1*(first[1] + (b)*space[1])))]
    ax1 = [float("%.4f" % (0.1*first[2])), float("%.4f" % (0.1*first[2]+0.1*(c)))]

    aa = [sag1, cor1, ax1]

    return aa

# just runs ^^ many times from choose in a vector
def a_plane_over_time_LOTS(parent_dir, folder, TEST, choices, all_time, how_many, size, here, speed):

    # first we prep the body by extracting the non extended cts
    # use True to extract only up till certain time
    # otherwise it will extract the whole thing
    pixelgrid4D = extract_4DCTpixels(folder, all_time, how_many)

    # next we will figure out values using the metadata in the first frame
    values = getallmetadata(folder, parent_dir, pixelgrid4D[0])

    for i in choices:
        choose = i
        print("\n\n\nRUNNING CHOOSE", choose)
        gif, saveable = a_plane_over_time(pixelgrid4D, TEST, choose, size, values)
        # gif, saveable = a_plane_over_time(folder, TEST, choose, all_time, how_many, size)
        name = typee + str(choose) + "slice"
        save_the_gif(gif, name, speed, here, saveable)

    return







###################################################################
# MAIN
###################################################################
aaa = time.perf_counter()


# DEFINE ALL INPUT FOLDERS WE NEED TO USE FOR EXTRACTION
timebody_folder = "/Users/felicialiu/Desktop/summer2022/coding/Cardiac 1.0 CE - 3"
typee = "AORTA"
TEST = "/Users/felicialiu/Desktop/summer2022/coding/centreline_files/" + typee + ".pth"
parent_dir = "/Users/felicialiu/Desktop/summer2022/coding/runthrough2"
speed = 80



# ***********
# PHASE ONE: generating all the gifs in a "gif" folder
# ***********

# make the gif folder - everything will be stored here
name = "gif"
here = make_dir(parent_dir, name)
# here = parent_dir + "/gif/"

# extract all gifs into folder "gif" in "runthrough2"
all_time = True # if True, extracted for every time interval
how_many = 10 # capped at 151 if all_time == False
size = 160 # aorta will use bigger 
# choices = choosing(2, 120, 8)
choices = [0, 10]
a_plane_over_time_LOTS(parent_dir, timebody_folder, TEST, choices, all_time, how_many, size, here, speed)
print("you need 'here' for next step:", here)








bbb = time.perf_counter()
###################################################################
# CLEAN UP
###################################################################

# just to mark the end of the script

print("\n\nWHOLE RUNTIME:")
tictoc(aaa, bbb)
print("done")

print("\n\n#######################################################")

