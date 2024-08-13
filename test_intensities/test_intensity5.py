'''
NEWEST INTENSITY DOCUMENT:
- actually using correct segments it generates now!!
- still need to define centrepoints
- makes individual folders in main parent "runthrough2"
- one folder "gifs" holds all vtp extracted gifs
- then each gif gets it's own folder of time intensities and everything!
- each plane's tic is extract for intensities
- from that, we get our average velocity
- then back track to flowrate!
'''

'''
NEW ADDITIONS:
- this will actually extract correct looking planes
- and the first centre is always chosen to be the centre of the frame
- might add new TIC interpretation stuff too? (ie. shape assumption?)
- and maybe a contrast improvement thing when moving to coronaries?
- and a manual size decider based on the vessel
- and a thing for the deciding if some frames are bad?
- knowing the range of bad clips and adding that to the plots to know noise
- and choosing how many planes and gifs to extract
- IT'LL BE MORE MANUAL!
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
from scipy.optimize import curve_fit


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
    
    # print(locations)
    if typee == "normal":
        new_array = pixel_copy(pixels)
    else: # working with grad here
        new_array = change1to3array(pixels)
    
    for i in locations:
        # new_array[i[0]][i[1]] = [255, 0, 0, 255]
        if i[0] < len(pixels) and i[1] < len(pixels[0]):
            new_array[i[0]][i[1]] = [255, 0, 0]
    
    for i in centres:
        # new_array[i[0]][i[1]] = [0, 255, 0, 255]
        if i[0] < len(pixels) and i[1] < len(pixels[0]):
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
    return float("%.4f" % ((sum/count)))

# interprolates in a function to make resulting function have more points between
# if L = [1,2], stretch = 4, then sL = [1, 1.25, 1.5, 1.75, 2]
def function_stretcher(Ls, stretch_num):
    wholer = []
    for L in Ls:
        sL = []
        for i in range(len(L)-1):
            adder = (L[i+1]-L[i])/stretch_num
            for j in range(stretch_num):
                sL.append(float("%.3f" % (L[i]+adder*j)))
        sL.append(float("%.3f" % (L[-1])))
        wholer.append(sL)
    spacer = 0.1/stretch_num
    # print("function stretched from", len(Ls[0]), "to", len(wholer[0]))
    return wholer, spacer

# finds average value of a list
def avgoflist_pos(L):
    sum = 0
    count = 0
    for i in range(len(L)):
        if L[i] > 0:
          sum = sum + L[i]
          count += 1
    return float("%.4f" % ((sum/count)))










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
    for i in range(1, end_range):

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

# chooses the frame size dependant on the type of vessel
def size_type_chooser(typee):
    aorta = ["AORTA"]
    branches = ["BCA", "LCC", "LSUB"]
    grafts = ["LIMA", "SVG1", "SVG2", "RA"]

    if typee in aorta:
        return 140, 12
    if typee in branches:
        return 50, 8
    if typee in grafts:
        return 50, 8
    else:
        return -1, 0

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
    # print("creating full centreline points dictionary...")
    # tic = time.perf_counter()

    file = open(path)
    ugly = file.read()
    new = segment_words(ugly)

    dict = {}

    for i in range(len(new)):
        s, wow = extracting_values(new[i])
        name = "point" + str(s)
        dict[name] = wow
    
    # print("done making dictionary.")
    # toc = time.perf_counter()
    # tictoc(tic, toc)
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
    if loc < 3 or loc > len(newpts)-4:
        print(loc, "is not a valid location. must be between 1 and", len(newpts)-4)
        return [0, 0, 0], [0, 0, 0], False
    
    lpos = newpts[loc-3]
    point = newpts[loc]
    mpos = newpts[loc+3]
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

# overlay the normal plane and voxels to extract plane of right intensities
# inputs, elogated pixelgrid, the centreline point, its normal vector, 
# and size of the output image array -> to get the image array (square)
def voxel_plane_extraction(pixel_grid, point, normal, size):
    
    intensities = np.array([[0]*size]*size)
    locs = bring_back_locs(point, normal, size)

    # then for each location already set, we'll create the new grid of pixel values
    for i in range(len(locs)):
        for j in range(len(locs[i])):
            intensities[i][j] = localintensity(locs[i][j], pixel_grid)

    return intensities

# to hopefully make the normals look better
def onethirdsizereduction(newpts):
    new = []
    count = 0
    for newpt in newpts:
        if count == 0:
            new.append(newpt)
        count += 1
        if count == 3:
            count = 0
    return new

# now to look at normal planes along these new axis things
def looking_along(pixel_grid, dict, size, values):
    newpts = position_remapper(dict, pixel_grid, values)
    # newpts = onethirdsizereduction(newpts)

    gif = []
    locas = []
    # for each of those points we want to see the normal plane
    for i in range(3, len(newpts)-3):
        print("mapping location", i)
        lpos = newpts[i-3]
        point = newpts[i]
        mpos = newpts[i+3]
        normal = [mpos[0] - lpos[0], mpos[1] - lpos[1] , mpos[2] - lpos[2]]
        intgrid = voxel_plane_extraction(pixel_grid, point, normal, size)
        locs = bring_back_locs(point, normal, size)
        locas.append(locs)
        gif.append(my_colourmap(intgrid))
        # printer2(intgrid)
    
    return gif, locas, newpts

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

# plots the little frame moving along the vessel
def plot_likesim(locas, newpts, pixel_grid, typee, here):

    # see what it looks like
    x = []
    y = []
    z = []
    for i in newpts:
        x.append(i[0])
        y.append(i[1])
        z.append(i[2])
    
    fig = plt.figure()
    images = []

    count = 0
    for i in locas:
        print('likesim plotting', count)
        count += 1
        ax = plt.axes(projection='3d')
        locs = i
        point1 = locs[0][0]
        point2 = locs[0][len(locs)-1]
        point3 = locs[len(locs)-1][len(locs)-1]
        point4 = locs[len(locs)-1][0]

        xx = [point1[0], point2[0], point3[0], point4[0], point1[0]]
        yy = [point1[1], point2[1], point3[1], point4[1], point1[1]]
        zz = [point1[2], point2[2], point3[2], point4[2], point1[2]]

        ax.plot3D(x, y, z, 'maroon')
        ax.plot3D(xx, yy, zz, 'green')
        ax.set_xlim3d(0, len(pixel_grid[0]))
        ax.set_ylim3d(0, len(pixel_grid[0][0]))
        ax.set_zlim3d(0, len(pixel_grid))
        ax.set_xlabel('sagital')
        ax.set_ylabel('coronal')
        ax.set_zlabel('axial')
        ax.set_title(typee + "line plot with moving normal plane")
        plt.savefig("/Users/felicialiu/Desktop/summer2022/coding/photos/ehh.png")
        # plt.show()
        data = im.open("/Users/felicialiu/Desktop/summer2022/coding/photos/ehh.png")
        images.append(data)
        os.remove("/Users/felicialiu/Desktop/summer2022/coding/photos/ehh.png")
    
    # save the gif!
    print("about to save")
    namee = here + typee + "lineplot.gif"
    images[0].save(namee, 
            save_all = True, 
            append_images = images[1:], 
            duration=60, 
            loop=0)

# just runs ^^ many times from choose in a vector
def a_plane_over_time_LOTS(parent_dir, folder, TEST, spacing, all_time, how_many, size, here, speed, typee):

    # first we prep the body by extracting the non extended cts
    # use True to extract only up till certain time
    # otherwise it will extract the whole thing
    pixelgrid4D = extract_4DCTpixels(folder, all_time, how_many)

    # next we will figure out values using the metadata in the first frame
    values = getallmetadata(folder, parent_dir, pixelgrid4D[0])

    # we can also get the full moving through the vessel effect?
    dict = init_path_file(TEST)
    pixel_grid = extend_3Dbody(pixelgrid4D[round(len(pixelgrid4D)/2)])
    gif, locas, newpts = looking_along(pixel_grid, dict, size, values)
    name = typee + "movingthrough"
    save_the_gif(gif, name, speed, parent_dir+"/")
    plot_likesim(locas, newpts, pixel_grid, typee, parent_dir+"/")

    # and lastly we'll make choices based off the spacing
    choices = choosing(3, len(dict)+2, spacing)

    for i in choices:
        choose = i
        print("\n\n\nRUNNING CHOOSE", choose)
        gif, saveable = a_plane_over_time(pixelgrid4D, TEST, choose, size, values)
        # gif, saveable = a_plane_over_time(folder, TEST, choose, all_time, how_many, size)
        name = typee + str(choose) + "slice"
        save_the_gif(gif, name, speed, here, saveable)

    return










###################################################################
# FUNCTION DEFINITIONS - removing bad gifs!
###################################################################

# for all choicses in removers, get rid of the gifs
# let's user manually get rid of bad ones
def remove_bad_gifs(typee, removers, here):
    for choose in removers:
        name = typee + str(choose) + "slice.gif"
        path = here + name
        os.remove(path)










###################################################################
# FUNCTION DEFINITIONS - TIC graph and point extraction!
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

# selects all the pixels around a small square
def small_square_around(center, size, pixels):
    locs = []
    for x in range(center[0]-round(size/2), center[0]+round(size/2), 1):
        for y in range(center[1]-round(size/2), center[1]+round(size/2), 1):
            locs.append([x, y])
    
    # the for each loc we find the intensity and report as sum
    sum = 0
    count = 0
    for loc in locs:
        if loc[0] < len(pixels) and loc[1] < len(pixels[0]):
            sum = sum + pixels[loc[0]][loc[1]]
            count += 1
    intensity = round(sum/count)

    if size < 3:
        thres = intensity-17
    else:
        thres = intensity-23
    return thres

# chooses the frame size dependant on the type of vessel
def frame_size_chooser(typee):
    aorta = ["AORTA"]
    branches = ["BCA", "LCC", "LSUB"]
    grafts = ["LIMA", "SVG1", "SVG2", "RA"]

    if typee in aorta:
        return 5
    if typee in branches:
        return 3
    if typee in grafts:
        return 2
    else:
        return -1

# GUI application to allow user to click on bolus centre
def clicker_man(pixels):
    
    # plots the graph
    fig, ax1 = plt.subplots()
    plt.imshow(pixels, cmap=plt.cm.gray)

    # mouse click function to store coordinates
    def onclick(event):
        # Only use event within the axes.
        if not event.inaxes == ax1:
            print('not in bounds, click again.')
            return

        ix, iy = event.xdata, event.ydata
        centre = [round(iy), round(ix)]
        coords3.append(centre)
        plt.close()
        plt.close()
        return centre
    
    # list to store coordinates
    coords3 = []
    print("\n**\nplease click on centre :")
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    centre = coords3[0]
    # print("centre :",centre)

    return centre

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

    if len(locations) == 0:
        print("*** REUSING OLD LOCATIONS")
        return []

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

    return [round((xcen+2*centres[-1][0])/3), round((ycen+2*centres[-1][1])/3)]

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

# looks for boundaries for next iteration
def find_dist(locations, centre):
    dists = []
    for loc in locations:
        dists.append(dist(centre, loc))
    return np.min(dists), np.max(dists)

# finds intensity for one plane using locations
def loc_pixels_intensity_extraction(locations, pixels):
    sum = 0
    count = 0
    for loc in locations:
        down = loc[0]
        side = loc[1]
        if down < len(pixels) and side < len(pixels[0]):
            sum = sum + pixels[down][side]
            count += 1
    return round(sum/count)

# takes in radius vectors and graphs with respect to 0.1 s time
def rad_plotter(TIC_vec, here, frame_size):
    print("making RAD plot...")
    tic = time.perf_counter()
    x = []
    # first make the x vector:
    for i in range(len(TIC_vec)):
        x.append(float("%.1f" % (i * (0.1))))
    
    title = "Time Radius Curve (TIC) of Slice - Case 1 CABG Project"
    
    plt.plot(x, TIC_vec, label='RAD')
    plt.xlabel("time of perfusion acquisition [s]")
    plt.ylabel("average radius of the vessel lumen [pixels]")
    plt.title(title)
    plt.legend()
    plt.xlim(0, max(x)+0.5)
    plt.ylim(0, frame_size)

    name = here + "RAD.png"
    plt.savefig(name)
    # plt.show()
    plt.clf()

    # while plotting, let's save the data to a txt file too!
    name1 = "time of perfusion acquisition [s]"
    thing1 = x
    name2 = "average radius of the vessel lumen [pixels]"
    thing2 = TIC_vec
    name = here + "/radresults.txt"
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
    
    print("plot created and saved.")
    tictoc(tic, toc)
    
    return x

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
    plt.xlim(0, (len(TIC_vec)+3)/10)
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
    
    print("plot created and saved.")
    tictoc(tic, toc)
    
    return x

# finds the average radius of one point
def this_slice_average_rad(cont, tot_centres, here, frame_size):

    total_rad = []
    for i in range(len(tot_centres)):
        centre = tot_centres[i]
        contpts = cont[i]

        dists = []
        for j in range(len(contpts)):
            
            dists.append(dist(centre, contpts[j]))
        rad = avgoflist(dists)
        total_rad.append(rad)
    
    # now we have radius plot time in the slice
    # we can plot that too!
    rad_plotter(total_rad, here, frame_size)

    # # we'll remove the top third of radii
    # # they're all on the big end
    # # also some of the smaller ones
    # for i in range(round(len(tot_centres)/3)):
    #     total_rad.remove(max(total_rad))
    # for i in range(round(len(tot_centres)/6)):
    #     total_rad.remove(min(total_rad))

    return avgoflist(total_rad)

# for one plane, runs the folder making tic generator
def the_whole_polarthing(gif_path, parent_dir, speed, frame_size):

    # first we make the directory to save everything to
    here = make_dir_gif(parent_dir, gif_path)
    
    # then we extract the gif
    pixel_grid = change3to1(extract_gif(gif_path))
    
    # the save the gif to this new folder
    save_the_gif(pixel_grid, "OG_GIF", speed, here)

    # flattener needs the body to be in 1
    flat = flattener(pixel_grid)
    # printer(flat)
    saver(flat, "flat", here)
    sized = round(len(flat)/2)

    centre = [sized, sized]

    # these you don't need to touch
    centres = [centre]
    tot_centres = []
    size = frame_size # for the square to find threshold
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
    contour_locsvec = []

    # centres will show only 3 and then disapear in that order

    for i in range(len(pixel_grid)):
    # for i in range(21):
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
        contour_locsvec.append(locations)

        # assign a new centre for the next round using these contour points
        new_cen = find_centre(locations, centres)
        centres.append(new_cen)
        tot_centres.append(new_cen)

        # we'll use the outside contour to find the min and max dist for next time
        min_dist, max_dist = find_dist(locations, centre)

        # now we can fill in the contour
        locations = filling_it_in(locations)
        holder = selected_shower(pixels, locations, centres, typee = "single", showw = False)
        reds.append(holder)

        # for the locations we want to extract the intensity and add to vector
        intensity = loc_pixels_intensity_extraction(locations, pixels)
        TIC_vec.append(intensity)

        # and we'll add centres till we reach 3 then always pop the first one out
        if len(centres) == 4:
            centres = centres[1:]


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

    avg_rad = this_slice_average_rad(contour_locsvec, tot_centres)
    # print("this slice average rad", avg_rad)

    return avg_rad

# for one plane, runs the folder making tic generator
# USES MOVING CENTRE AND MANUAL SET STARTING CENTRE
def the_whole_polarthing_mmm(gif_path, parent_dir, speed, frame_size):

    # first we make the directory to save everything to
    here = make_dir_gif(parent_dir, gif_path)
    
    # then we extract the gif
    pixel_grid = change3to1(extract_gif(gif_path))
    
    # the save the gif to this new folder
    save_the_gif(pixel_grid, "OG_GIF", speed, here)

    # flattener needs the body to be in 1
    flat = flattener(pixel_grid)
    # printer(flat)
    saver(flat, "flat", here)
    # sized = round(len(flat)/2)

    # manually gets our centre
    centre = clicker_man(pixel_grid[0])
    print("centre :",centre)

    # these you don't need to touch
    centres = [centre]
    tot_centres = []
    size = frame_size # for the square to find threshold
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
    contour_locsvec = []

    # centres will show only 3 and then disapear in that order
    for i in range(len(pixel_grid)):
    # for i in range(49):
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
        contour_locsvec.append(locations)

        # assign a new centre for the next round using these contour points
        new_cen = find_centre(locations, centres)
        centres.append(new_cen)
        tot_centres.append(new_cen)
        centre = new_cen

        # we'll use the outside contour to find the min and max dist for next time
        min_dist, max_dist = find_dist(locations, centre)

        # now we can fill in the contour
        locations = filling_it_in(locations)
        holder = selected_shower(pixels, locations, centres, typee = "single", showw = False)
        reds.append(holder)

        # for the locations we want to extract the intensity and add to vector
        intensity = loc_pixels_intensity_extraction(locations, pixels)
        TIC_vec.append(intensity)

        # and we'll add centres till we reach 3 then always pop the first one out
        if len(centres) == 4:
            centres = centres[1:]
        
        if i >50:
            save_the_gif(np.array(cont), "CONTOUR", speed, here)


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

    avg_rad = this_slice_average_rad(contour_locsvec, tot_centres)
    # print("this slice average rad", avg_rad)

    return avg_rad

# stitches two lists together
def stitch_together(one, two):
    three = []
    for i in one:
        three.append(i)
    for i in two:
        three.append(i)
    return three

# runs the middle part of whole_polarthing
def mini_polar_runner(centre, frame_size, pixel_grid):

    # these you don't need to touch
    centres = [centre]
    tot_centres = []
    size = frame_size # for the square to find threshold
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
    contour_locsvec = []

    # centres will show only 3 and then disapear in that order
    for i in range(len(pixel_grid)):
    # for i in range(22):
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

        if locations == []:
            # that means the last place had no locations
            # we'll use 3 ago's contour locations
            if len(contour_locsvec) > 2:
                locations = contour_locsvec[-3]
            else:
                locations = contour_locsvec[-1]
        else:
            locations = contouring2(locations, centre) # will prevent weird breaks
        heap = selected_shower(pixels, locations, centres, typee = "single", showw = False)
        cont.append(heap)
        contour_locsvec.append(locations)

        # assign a new centre for the next round using these contour points
        new_cen = find_centre(locations, centres)
        centres.append(new_cen)
        tot_centres.append(new_cen)
        centre = new_cen

        # we'll use the outside contour to find the min and max dist for next time
        min_dist, max_dist = find_dist(locations, centre)

        # now we can fill in the contour
        locations = filling_it_in(locations)
        holder = selected_shower(pixels, locations, centres, typee = "single", showw = False)
        reds.append(holder)

        # for the locations we want to extract the intensity and add to vector
        intensity = loc_pixels_intensity_extraction(locations, pixels)
        TIC_vec.append(intensity)

        # and we'll add centres till we reach 3 then always pop the first one out
        if len(centres) == 4:
            centres = centres[1:]
    
    return reds, cont, dots, starter, TIC_vec, contour_locsvec, tot_centres

# for one plane, runs the folder making tic generator
# USES MOVING CENTRE AND MANUAL SET STARTING CENTRE
# THIS ONE ALSO starts in the middle and stiches the halves back together
def the_whole_polarthing_mhmm(gif_path, parent_dir, speed, frame_size, size):

    # first we make the directory to save everything to
    here = make_dir_gif(parent_dir, gif_path)
    
    # then we extract the gif
    pixel_grid = change3to1(extract_gif(gif_path))
    
    # the save the gif to this new folder
    save_the_gif(pixel_grid, "OG_GIF", speed, here)

    # flattener needs the body to be in 1
    flat = flattener(pixel_grid)
    # printer(flat)
    saver(flat, "flat", here)
    # sized = round(len(flat)/2)

    # manually gets our centre
    mid = round(len(pixel_grid)/2)
    centre = clicker_man(pixel_grid[mid])
    print("centre :",centre)

    # now we run the mini one twice
    # first half will be reversed
    # second half will be fine
    # then we stitch them back together
    # then we save it all
    pixel_grid1 = reverse_order(pixel_grid[:mid])
    pixel_grid2 = pixel_grid[mid:]
    reds1, cont1, dots1, starter1, TIC_vec1, contour_locsvec1, tot_centres1 = mini_polar_runner(centre, frame_size, pixel_grid1)
    reds2, cont2, dots2, starter2, TIC_vec2, contour_locsvec2, tot_centres2 = mini_polar_runner(centre, frame_size, pixel_grid2)

    reds = stitch_together(reverse_order(reds1), reds2)
    cont = stitch_together(reverse_order(cont1), cont2)
    dots = stitch_together(reverse_order(dots1), dots2)
    starter = stitch_together(reverse_order(starter1), starter2)
    TIC_vec =  stitch_together(reverse_order(TIC_vec1), TIC_vec2)
    contour_locsvec = stitch_together(reverse_order(contour_locsvec1), contour_locsvec2)
    tot_centres = stitch_together(reverse_order(tot_centres1), tot_centres2)
    
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

    avg_rad = this_slice_average_rad(contour_locsvec, tot_centres, here, size)
    # print("this slice average rad", avg_rad)

    return avg_rad

# main for this step
def run_tics_for_all_gifs(parent_dir, speed, frame_size, size):
    gif_folder_path = parent_dir + "/gif/"
    names = os.listdir(gif_folder_path)
    
    rads = []
    for name in names:

        if name == ".DS_Store":
            continue
    
        gif_path = gif_folder_path + name
        print("*******\nWORKING ON", name)
        rad = the_whole_polarthing_mhmm(gif_path, parent_dir, speed, frame_size, size)
        rads.append(rad)
    
    # to account for some areas giving larger radius
    print(rads)
    # rads.remove(max(rads))
    # rads.remove(max(rads))
    # rads.remove(max(rads))
    rrad = avgoflist(rads)
    return rrad









###################################################################
# FUNCTION DEFINITIONS - doing the velocity!
###################################################################

# takes in parent folder and figures out the correct folder order
# gives those folder names as a list
def folder_opener(typee, parent_dir):
    names = os.listdir(parent_dir)
    hold = []
    for name in names:
        if typee in name:
            if "movingthrough.gif" not in name:
                if "lineplot.gif" not in name:
                    new = name.replace(typee, "")
                    new = new.replace("slice", "")
                    new = int(new)
                    hold.append(new)
    hold.sort()
    goodnames = []
    nose = []
    for i in hold:
        goodnames.append(parent_dir + "/" + typee + str(i) + "slice/")
        nose.append(typee + str(i) + "slice")
    
    return goodnames, nose

# takes in one gif folder opens and extracts the TIC_vector
# returns as a list
def TIC_vecout(path):
    new_path = path + "plotresults.txt"
    horrible = open(new_path, "r").readlines()
    
    index1 = horrible.index('time of perfusion acquisition [s]\n')
    index = horrible.index('intensity of the pixels [grayscale]\n')
    times = []
    for i in range(index1+1, index-1, 1):
        num = float((horrible[i]).replace("\n", ""))
        times.append(num)

    TIC_vec = []
    for i in range(index+1, len(horrible), 1):
        num = int((horrible[i]).replace("\n", ""))
        TIC_vec.append(num)
    
    return times, TIC_vec

# opens folder and extracts all TIC_vecs
def preparing_TICS(typee, parent_dir):

    # first from the directory we get all the folder names
    paths_list, names = folder_opener(typee, parent_dir)

    # then for each path, we extract it's vector and add it to a list
    ALL_vecs = []
    ALL_times = []
    for path in paths_list:
        times, TIC_vec = TIC_vecout(path)
        ALL_vecs.append(TIC_vec)
        ALL_times.append(times)
    
    # plot_ALL_tics(typee, ALL_times, ALL_vecs, names)
    return ALL_vecs, names

# plots all tic vecs in a list
def plot_ALL_tics(spacer, colours, typee, ALL_vecs, names, here, use="", save=False, showw=True):
    # print("graphing ALL TICs ...")
    title = typee + ": All Intensity Time Curves (TICs) - Case 1 CABG Project"

        
    # making the ultimate plot
    for i in range(len(ALL_vecs)):
        x = []
        # first make the x vector:
        for j in range(len(ALL_vecs[i])):
            x.append( float("%.3f" % (j * (spacer))))
        plt.plot(x, ALL_vecs[i], label=names[i], color=colours[i])

    plt.xlabel("time of perfusion acquisition [s]")
    plt.ylabel("intensity of the pixels [grayscale]")
    plt.title(title)
    plt.legend()
    # plt.xlim(0, 16)
    plt.ylim(0, 255)
    plt.grid()
    plt.axes

    if save == True:
        name = here + "/" + use + '.png'
        plt.savefig(name)

    if showw == True:
        plt.show()
    plt.clf()

    # print("that's it.\n")
    return
    
# plots all tic vecs in a list
def plot_two(typee, vec1, vec2):
    print("graphing two TICs ...")
    title = typee + ": Two Intensity Time Curves (TICs) - Case 1 CABG Project"
    
    x = []
    # first make the x vector:
    for i in range(len(vec1)):
        x.append( float("%.1f" % (i * (0.1))))
    
    xx = []
    # first make the x vector:
    for i in range(len(vec2)):
        xx.append( float("%.1f" % (i * (0.1))))
    
    plt.plot(x, vec1, label="first one")
    plt.plot(xx, vec2, label="second one")
    plt.xlabel("time of perfusion acquisition [s]")
    plt.ylabel("intensity of the pixels [grayscale]")
    plt.title(title)
    plt.legend()
    # plt.xlim(0, 16)
    plt.ylim(0, 255)
    plt.grid()

    plt.show()

    print("that's it.\n")
    return

# makes TICs vec smoother by averaging
def smooth_with_neighbours(vec, how_much):
    new = []
    for i in range(len(vec)):
        sum = 0
        count = 0
        for j in range(i-how_much, i+how_much+1, 1):
            if j >= 0:
                if j < len(vec):
                    sum = sum + vec[j]
                    count += 1
        new.append(round(sum/count))
    return new

# applies the neighbour smoothing to all TIC_vecs
def smooth_all(ALL_vecs, how_much):
    smooths = []
    for i in range(len(ALL_vecs)):
        vec = ALL_vecs[i]
        for h in range(5):
            vec = smooth_with_neighbours(vec, how_much)
        smooths.append(vec)
    return smooths

# finds distance between two points in 3D
def distance3D(point1, point2):
    x = point2[0] - point1[0]
    y = point2[1] - point1[1]
    z = point2[2] - point1[2]
    return math.sqrt(x**2 + y**2 + z**2)

# figures out the real length using the pathpoints
def find_rlength(TEST):
    dict = init_path_file(TEST)
    linepts = []
    for key in dict:
        linepts.append(dict[key][0])
    
    nnewpts = []
    for newpt in linepts:
        nnewpts.append([newpt[0], newpt[1], newpt[2]*3+1])

    sum = 0
    for i in range(len(linepts)-1):
        sum = sum + distance3D(linepts[i], linepts[i+1])
    return round(sum*10)

# figures out the vessel length based on the slice number
# need real length in
def name_to_length_converter(names, typee, TEST):
    
    # this will give us the points length
    dict = init_path_file(TEST)
    plength = len(dict)
    rlength = find_rlength(TEST)
    print("real length is:", rlength, "mm")
    
    lengths = []
    for name in names:
        num = name.replace(typee, "")
        num = num.replace("slice","")
        num = int(num)
        lengths.append(float("%.4f" % (rlength*(num/plength))))
    return lengths

# finds error between all terms
def calculate_error(one, two):
    error = 0
    for i in range(len(one)):
        error = error + abs(one[i]-two[i])**2
    return error

# reports the time to which the second delays the first
def stalker(first, second, spacer):
    times = []
    errors = []
    for i in range(round(len(first)*3/4)):
        times.append(float("%.4f" % (i * (spacer))))
        # now we wanna find the error
        one = first[:len(first)-i]
        two = second[i:]
        # plot_two("typee", one, two)
        errors.append(calculate_error(one, two))

    # for tracking the minimum error
    minerr = np.min(errors)
    for i in range(len(errors)-1, -1, -1):
        if errors[i] == minerr:
            return times[i]

    return False

# figures out the time displacement based on the error minimised
def vecs_to_times_converter(ALL_vecs, spacer):
    times = [0.0]
    for i in range(1, len(ALL_vecs)):
        times.append(stalker(ALL_vecs[0], ALL_vecs[i], spacer))
    return times

# ^^ same but only to the next adjacent frame
def vecs_to_times_converter_neighbour(ALL_vecs, spacer):
    times = [0.0]
    for i in range(0, len(ALL_vecs)-1):
        times.append(stalker(ALL_vecs[i], ALL_vecs[i+1], spacer))
    
    # THEN EACH ELEMENT SHOULD BE THE SUM OF THE TIMES BEFORE IT
    times = np.array(times)
    new_times = []
    for i in range(1, len(times)+1):
        new_times.append(float("%.4f" % np.ndarray.sum(times[:i])))

    return new_times

# just to make sure it actually looks better?
# correct all the ALL_vecs
def plot_sanity_check_on_times(times, ALL_vecs, spacer):
    new = []
    for i in range(len(times)):
        cut = int(times[i]/spacer)
        ahh = ALL_vecs[i][cut:]
        # print(ahh, len(ahh))
        # plot_two("typee", ahh, ALL_vecs[i])
        new.append(ahh)
    return new

# creates a colours vector
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
    # first pass : blue to purple
    for n in range(br):
        list.append([r, g, b])
        sub = round(150/br)
        sub2 = round((150/br)/2)
        b = max(0, b-sub2)
        r = min(255, r+sub)

    # printer([list])

    return list
def convert_rgbtohex(L):
    hex = []
    for three in L:
        use = (three[0], three[1], three[2])
        hex.append('#%02x%02x%02x' % use)
    return hex

# regression velocity version
def estimate_coef(x, y):
    # number of observations/points
    n = np.size(x)
  
    # mean of x and y vector
    m_x = np.mean(x)
    m_y = np.mean(y)
  
    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x
  
    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
  
    return (b_0, b_1)
def plot_regression_line(namee, x, y, b, save, here, showw=True):
    # plotting the actual points as scatter plot
    plt.scatter(x, y, color = "m",
               marker = "o", s = 30)
  
    # predicted response vector
    y_pred = b[0] + b[1]*x
  
    # plotting the regression line
    plt.plot(x, y_pred, color = "g")
  
    # putting labels
    plt.xlabel("time delay of the bolus with 0mm as 0s [s]")
    plt.ylabel("lengths over the vessel with earliest frame as proximal [mm]")
    plt.title(namee + ": Time Delays Over the Length of the Vessel")
    plt.xlim(0,)
    plt.ylim(0,)
    plt.grid()
    if save == True:
        name = here + "/" + namee + "reg_velocity.png"
        plt.savefig(name)
    # function to show plot
    if showw == True:
        plt.show()
    plt.clf()
    return

# crop all vectors by the same point
def time_crop_vecs(ALL_vecs, timee):
    new_vecs = []
    for vec in ALL_vecs:
        new_vecs.append(vec[timee:])
    return new_vecs

# approximates any data with gaussian distribution
def gauss_it(ALL_vecs, spacer):

    # Let's create a function to model and create data
    def gauss(x, a, x0, sigma):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))
  
    news = []
    for i in range(len(ALL_vecs)):
        x = []
        # first make the x vector:
        for j in range(len(ALL_vecs[i])):
            x.append( float("%.3f" % (j * (spacer))))
        
        # now we want to find the gauss approxmiate
        popt, pcov = curve_fit(gauss, x, ALL_vecs[i])
        ym = gauss(x, popt[0], popt[1], popt[2])
        news.append(ym)
    return news

# make some plots?
def plot_lengths_times_velocities(namee, lengths, times, here, save=False, showw=True):

    # plot of x:lengths and y:times
    plt.scatter(lengths, times, label="time delay")
    plt.xlabel("lengths over the vessel with earliest frame as proximal [mm]")
    plt.ylabel("time delay of the bolus with 0mm as 0s [s]")
    plt.title(namee + ": Time Delays Over the Length of the Vessel")
    plt.legend()
    plt.xlim(0,)
    plt.ylim(0,)
    plt.grid()
    if save == True:
        name = here + "/"+ namee +"length_time.png"
        plt.savefig(name)
        # while plotting, let's save the data to a txt file too!
        name1 = "lengths over the vessel with earliest frame as proximal [mm]"
        thing1 = lengths
        name2 = "time delay of the bolus with 0mm as 0s [s]"
        thing2 = times
        name = here + "/"+ namee +"length_time.txt"
        textfile = open(name, "w")
        textfile.write(name1 + "\n")
        for element in thing1:
            textfile.write(str(element) + "\n")
        textfile.write("\n")
        textfile.write(name2 + "\n")
        for element in thing2:
            textfile.write(str(element) + "\n")
        textfile.close()

    if showw == True:
        plt.show()
    plt.clf()

    # first we can make a velocity vector
    # and with it, find average lengths 
    avglengths = []
    vels = []
    avglengths2 = []
    vels2 = []
    avglengths3 = []
    vels3 = []
    for i in range(len(lengths)-1):
        atime = times[i]
        btime = times[i+1]
        alen = lengths[i]
        blen = lengths[i+1]
        if (btime-atime) <= 0:
            use = 0.3
            avglengths2.append(float("%.4f" % ((alen+blen)/2)))
            vels2.append(float("%.4f" % ((blen-alen)/use)))
        else: 
            use = (btime-atime)
            avglengths.append(float("%.4f" % ((alen+blen)/2)))
            vels.append(float("%.4f" % ((blen-alen)/use)))
        avglengths3.append(float("%.4f" % ((alen+blen)/2)))
        vels3.append(float("%.4f" % ((blen-alen)/use)))
    
    avgvel = avgoflist(vels)

    x = np.array(times)
    y = np.array(lengths)
    b = estimate_coef(x, y)
    plot_regression_line(namee, x, y, b, save, here, showw)

    # plot of x:lengths and y:velocities 
    plt.scatter(avglengths, vels, label="velocities")
    plt.scatter(avglengths2, vels2, label="bad velocities?")
    plt.xlabel("lengths over the vessel with earliest frame as proximal [mm]")
    plt.ylabel("velocity [mm/s]")
    plt.title(namee + ": Average Cross-Sectional Velocity Over the Length of the Vessel")
    plt.legend()
    plt.xlim(0,)
    plt.ylim(0,)
    plt.grid()
    if save == True:
        name = here + "/"+ namee +"length_velocity.png"
        plt.savefig(name)
        # while plotting, let's save the data to a txt file too!
        name1 = "lengths over the vessel with earliest frame as proximal [mm]"
        thing1 = avglengths3
        name2 = "velocity [mm/s]"
        thing2 = vels3
        name = here + "/"+ namee +"length_velocity.txt"
        textfile = open(name, "w")
        textfile.write(name1 + "\n")
        for element in thing1:
            textfile.write(str(element) + "\n")
        textfile.write("\n")
        textfile.write(name2 + "\n")
        for element in thing2:
            textfile.write(str(element) + "\n")
        textfile.write("\n")
        textfile.write("average good velocities: " + str(avgvel) + " [mm/s]")
        textfile.close()
    if showw == True:
        plt.show()
    plt.clf()

    print(namee +"average good velocities: " + str(float("%.4f" % (avgvel))) + " [mm/s]")
    print(namee +"regression velocity: " + str(float("%.4f" % (b[1]))) + " [mm/s]")

    return avgvel, round(b[1])

# full velocity calculations
def running_velocity_extraction(colours, names, ALL_vecs, spacer, TEST, parent_dir, typee, save=False, showw=True):

    print("** NO SHAPE APPROXIMATION **")
    here = parent_dir + "/NO_APPROX"

    plot_ALL_tics(spacer, colours, typee, ALL_vecs, names, here, "ALL_TICS", save, showw)

    # smoothing the TICs to be less aggressive
    how_much = 2 # how many of each side smoothing factor
    ALL_vecs = smooth_all(ALL_vecs, how_much)
    plot_ALL_tics(spacer, colours, typee, ALL_vecs, names, here, "SMOOTH_TICS", save, showw)

    # makes a vector for lengths
    lengths = name_to_length_converter(names, typee, TEST)
    # ALL_vecs will be extended a bit to make them more continuous
    stretch_num = 5
    ALL_vecs, spacer = function_stretcher(ALL_vecs, stretch_num)

    # makes a vectors for times - all compared to first frame
    times = vecs_to_times_converter(ALL_vecs, spacer)
    # print(times)
    new = plot_sanity_check_on_times(times, ALL_vecs, spacer)
    plot_ALL_tics(spacer, colours, typee, new, names, here, "ffSHIFTED_TICS", save, showw)
    v1, v2 = plot_lengths_times_velocities("ff", lengths, times, here, save, showw)

    # new way to make vectors for times - compared to previous frame only? 
    times = vecs_to_times_converter_neighbour(ALL_vecs, spacer)
    # print(times)
    new = plot_sanity_check_on_times(times, ALL_vecs, spacer)
    plot_ALL_tics(spacer, colours, typee, new, names, here, "nfSHIFTED_TICS", save, showw)
    v3, v4 = plot_lengths_times_velocities("nf", lengths, times, here, save, showw)

    v = [v1, v2, v3, v4]
    

    return avgoflist_pos(v)

# full velocity calculations
def running_velocity_extraction_GAUSS(colours, names, ALL_vecs, spacer, TEST, parent_dir, typee, save=False, showw=True):

    print("** GAUSSIAN APPROXIMATION **")
    here = parent_dir  + "/GAUSS_APPROX"

    plot_ALL_tics(spacer, colours, typee, ALL_vecs, names, here, "ALL_TICS", save, showw)

    # this is instead of smoothing, the gaussian being applied
    ALL_vecs = gauss_it(ALL_vecs, spacer)
    plot_ALL_tics(spacer, colours, typee, ALL_vecs, names, here, "GAUSS_TICS", save, showw)

    # makes a vector for lengths
    lengths = name_to_length_converter(names, typee, TEST)
    # ALL_vecs will be extended a bit to make them more continuous
    stretch_num = 5
    ALL_vecs, spacer = function_stretcher(ALL_vecs, stretch_num)

    # makes a vectors for times - all compared to first frame
    times = vecs_to_times_converter(ALL_vecs, spacer)
    # print(times)
    new = plot_sanity_check_on_times(times, ALL_vecs, spacer)
    plot_ALL_tics(spacer, colours, typee, new, names, here, "ffSHIFTED_TICS", save, showw)
    v1, v2 = plot_lengths_times_velocities("ff", lengths, times, here, save, showw)

    # new way to make vectors for times - compared to previous frame only? 
    times = vecs_to_times_converter_neighbour(ALL_vecs, spacer)
    # print(times)
    new = plot_sanity_check_on_times(times, ALL_vecs, spacer)
    plot_ALL_tics(spacer, colours, typee, new, names, here, "nfSHIFTED_TICS", save, showw)
    v3, v4 = plot_lengths_times_velocities("nf", lengths, times, here, save, showw)

    v = [v1, v2, v3, v4]
    

    return avgoflist_pos(v)

# does both ^^
def run_both_velocities(TEST, parent_dir, typee, save=False, showw=True):

    # extracting all TICs
    ALL_vecs, names = preparing_TICS(typee, parent_dir)
    spacer = 0.1
    
    # get colours based on number needed
    rgb = return_colours(len(names))
    colours = convert_rgbtohex(rgb)

    # apply length cropper ask?
    if showw == True:
        plot_ALL_tics(spacer, colours, typee, ALL_vecs, names, here, "ALL_TICS", False, showw)
        cropper = input("enter time crop amount [int] or ['no'] :  ")
        if cropper == "no":
            pass
        else:
            ALL_vecs = time_crop_vecs(ALL_vecs, round(int(cropper)/spacer))
    else:
        pass
    
    print("\n\n")
    vv = running_velocity_extraction(colours, names, ALL_vecs, spacer, TEST, parent_dir, typee, save, showw)
    print("\n")
    vvv = running_velocity_extraction_GAUSS(colours, names, ALL_vecs, spacer, TEST, parent_dir, typee, save, showw)

    v = [vv, vvv]
    return avgoflist(v)
      
# opens folder and extracts all TIC_vecs
def preparing_TICS2(typee, parent_dir):

    # first from the directory we get all the folder names
    paths_list, names = folder_opener(typee, parent_dir)

    # then for each path, we extract it's vector and add it to a list
    ALL_vecs = []
    ALL_times = []
    for path in paths_list:
        times, TIC_vec = TIC_vecout(path)
        ALL_vecs.append(TIC_vec)
        ALL_times.append(times)
    
    
    return ALL_vecs, ALL_times, names

# plots all tic vecs in a list
def plot_ALL_tics2(colours, typee, ALL_times, ALL_vecs, names, here, use="", save=False, showw=True):
    # print("graphing ALL TICs ...")
    title = typee + ": All Intensity Time Curves (TICs) - Case 1 CABG Project"

    # making the ultimate plot
    for i in range(len(ALL_vecs)):
        plt.plot(ALL_times[i], ALL_vecs[i], label=names[i], color=colours[i])

    plt.xlabel("time of perfusion acquisition [s]")
    plt.ylabel("intensity of the pixels [grayscale]")
    plt.title(title)
    plt.legend()
    # plt.xlim(0, 16)
    plt.ylim(0, 255)
    plt.grid()
    plt.axes

    if save == True:
        name = here + "/" + use + '.png'
        plt.savefig(name)

    if showw == True:
        plt.show()
    plt.clf()

    # print("that's it.\n")
    return
 
# plots one tic vec in a list and returns indices to crop
def plot_two_cut(time, vec):

    fig, ax1 = plt.subplots()
    title = "Cropping Intensity Time Curves (TICs) - Case 1 CABG Project"
    plt.plot(time, vec, label="TIC", color="black")
    plt.xlabel("time of perfusion acquisition [s]")
    plt.ylabel("intensity of the pixels [grayscale]")
    plt.title(title)
    plt.legend()
    plt.xlim(0, len(time)+1)
    plt.ylim(0, 255)
    plt.grid()

    # mouse click function to store coordinates
    def onclick(event):
        # Only use event within the axes.
        if not event.inaxes == ax1:
            if len(coords3) == 2:
                print('done this crop.')
                plt.close()
                plt.close()
                return coords3
            print("more clicks needed. continue.")
            return
        ix, iy = event.xdata, event.ydata
        coords3.append(float("%.3f" % (ix)))
        
        if len(coords3) == 3:
            coords3.remove(coords3[0])

        if len(coords3) > 0:
            for ix in coords3:
                ax1.axvline(x=ix, ymin=0, ymax=1, color = "red")
                fig.canvas.draw_idle()
        
        coords = []
        for c in coords3:
            coords.append(c)
        coords.sort()
        print("current crop zone : ", coords)

        return
    
    # list to store coordinates
    coords3 = []
    print("\n**\nchoose boundaries to crop TIC : [click outside bounds to end]")
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    ind = []
    for ix in coords3:
        ind.append(round(ix*10))
    ind.sort()
    return ind

# plots one tic vec in a list
def plot_two_no_cut(time, vec):
    fig, ax1 = plt.subplots()
    title = "Cropping Intensity Time Curves (TICs) - Case 1 CABG Project"
    plt.plot(time, vec, label="TIC", color="black")
    plt.xlabel("time of perfusion acquisition [s]")
    plt.ylabel("intensity of the pixels [grayscale]")
    plt.title(title)
    plt.legend()
    plt.xlim(0, len(time)+1)
    plt.ylim(0, 255)
    plt.grid()

    # mouse click function to store coordinates
    def onclick(event):
        # Only use event within the axes.
        if not event.inaxes == ax1:
            plt.close()
            plt.close()
            return
        return
    
    # list to store coordinates
    print("\n**\ndisplaying cropped. [click outside bounds to close]")
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    return

# crops the vec and makes sure it looks right
def value_cropper(time, vec):
    # newtime = []
    # for i in time:
    #     newtime.append(i)
    # newvec = []
    # for i in vec:
    #     newvec.append(i)

    crop = plot_two_cut(time, vec)
    newtime = time[crop[0]+1 : crop[1]]
    newvec = vec[crop[0]+1 : crop[1]]
    plot_two_no_cut(newtime, newvec)

    return newtime, newvec, crop

# let's user define bounds to crop all images
def timecrop_bad(ALL_times, ALL_vecs):
    new_times = []
    new_vecs = []
    crops = []
    for i in range(len(ALL_times)):
        newtime, newvec, crop = value_cropper(ALL_times[i], ALL_vecs[i])
        new_times.append(newtime)
        new_vecs.append(newvec)
        crops.append(crop)
    return new_times, new_vecs, crops

# reports the time to which the second delays the first
def stalker_SMALL(first, second, spacer):
    times = []
    errors = []
    for i in range(round(len(first)*3/4)):
        times.append(float("%.4f" % (i * (spacer))))
        # now we wanna find the error
        one = first[:len(first)-i]
        two = second[i:]
        # plot_two("typee", one, two)
        errors.append(calculate_error(one, two))

    # for tracking the minimum error
    minerr = np.min(errors)
    for i in range(len(errors)-1, -1, -1):
        if errors[i] == minerr:
            return times[i]

    return False

# figures out the time displacement based on the error minimised
def vecs_to_times_converter_SMALL(ALL_vecs, spacer):
    times = [0.0]
    for i in range(1, len(ALL_vecs)):

        # each time we do this, need to make sure they start at the same value and have overlap

        times.append(stalker(ALL_vecs[0], ALL_vecs[i], spacer))
    return times

# full velocity calculations
def running_velocity_extraction_SMALL(crops, colours, names, ALL_vecs, ALL_times, TEST, parent_dir, typee, save=False, showw=True):

    print("** NO SHAPE APPROXIMATION **")
    here = parent_dir + "/NO_APPROX"

    plot_ALL_tics2(colours, typee, ALL_times, ALL_vecs, names, here, "ALL_TICS", save, showw)
      
    # smoothing the TICs to be less aggressive
    how_much = 2 # how many of each side smoothing factor
    ALL_vecs = smooth_all(ALL_vecs, how_much)
    plot_ALL_tics2(colours, typee, ALL_times, ALL_vecs, names, here, "SMOOTH_TICS", save, showw)
      
    # makes a vector for lengths
    lengths = name_to_length_converter(names, typee, TEST)
    # ALL_vecs will be extended a bit to make them more continuous
    stretch_num = 5
    ALL_vecs, spacer1 = function_stretcher(ALL_vecs, stretch_num)
    ALL_times, spacer2 = function_stretcher(ALL_times, stretch_num)
    # plot_ALL_tics2(colours, typee, ALL_times, ALL_vecs, names, here, "ALL_TICS", save, showw)
    
    # makes a vectors for times - all compared to first frame
    times = vecs_to_times_converter_SMALL(ALL_vecs, spacer1, crops)
    # print(times)
    # new = plot_sanity_check_on_times(times, ALL_vecs, spacer)
    plot_ALL_tics2(colours, typee, ALL_times, ALL_vecs, names, here, "ALL_TICS", False, showw)
    v1, v2 = plot_lengths_times_velocities("ff", lengths, times, here, save, showw)

    # # new way to make vectors for times - compared to previous frame only? 
    # times = vecs_to_times_converter_neighbour(ALL_vecs, spacer)
    # # print(times)
    # new = plot_sanity_check_on_times(times, ALL_vecs, spacer)
    # plot_ALL_tics(spacer, colours, typee, new, names, here, "nfSHIFTED_TICS", save, showw)
    # v3, v4 = plot_lengths_times_velocities("nf", lengths, times, here, save, showw)

    # v = [v1, v2, v3, v4]
    

    # return avgoflist_pos(v)

# full velocity calculations
def running_velocity_extraction_GAUSS_SMALL(colours, names, ALL_vecs, ALL_times, TEST, parent_dir, typee, save=False, showw=True):

    print("** GAUSSIAN APPROXIMATION **")
    here = parent_dir  + "/GAUSS_APPROX"

    plot_ALL_tics2(colours, typee, ALL_times, ALL_vecs, names, here, "ALL_TICS", False, showw)
        
    # this is instead of smoothing, the gaussian being applied
    ALL_vecs = gauss_it(ALL_vecs, spacer)
    plot_ALL_tics2(colours, typee, ALL_times, ALL_vecs, names, here, "ALL_TICS", False, showw)
     
    # makes a vector for lengths
    lengths = name_to_length_converter(names, typee, TEST)
    # ALL_vecs will be extended a bit to make them more continuous
    stretch_num = 5
    ALL_vecs, spacer1 = function_stretcher(ALL_vecs, stretch_num)
    ALL_times, spacer2 = function_stretcher(ALL_times, stretch_num)

    # makes a vectors for times - all compared to first frame
    times = vecs_to_times_converter(ALL_vecs, spacer)
    # print(times)
    new = plot_sanity_check_on_times(times, ALL_vecs, spacer)
    plot_ALL_tics(spacer, colours, typee, new, names, here, "ffSHIFTED_TICS", save, showw)
    v1, v2 = plot_lengths_times_velocities("ff", lengths, times, here, save, showw)

    # new way to make vectors for times - compared to previous frame only? 
    times = vecs_to_times_converter_neighbour(ALL_vecs, spacer)
    # print(times)
    new = plot_sanity_check_on_times(times, ALL_vecs, spacer)
    plot_ALL_tics(spacer, colours, typee, new, names, here, "nfSHIFTED_TICS", save, showw)
    v3, v4 = plot_lengths_times_velocities("nf", lengths, times, here, save, showw)

    v = [v1, v2, v3, v4]
    

    return avgoflist_pos(v)

# does both ^^
def run_both_velocities_SMALL(TEST, parent_dir, typee, save=False, showw=True):

    # extracting all TICs
    ALL_vecs, ALL_times, names = preparing_TICS2(typee, parent_dir)
    spacer = 0.1

    # get colours based on number needed
    rgb = return_colours(len(names))
    colours = convert_rgbtohex(rgb)
    plot_ALL_tics2(colours, typee, ALL_times, ALL_vecs, names, here, "ALL_TICS", False, showw)
        
    # we're going to crop the ends off the function depending 
    # on what's pressed and draw black verticle lines
    ALL_times, ALL_vecs, crops = timecrop_bad(ALL_times, ALL_vecs)
    plot_ALL_tics2(colours, typee, ALL_times, ALL_vecs, names, here, "ALL_TICS", False, showw)
     
    print("\n\n")
    vv = running_velocity_extraction_SMALL(crops, colours, names, ALL_vecs, ALL_times, TEST, parent_dir, typee, save, showw)
    # print("\n")
    # vvv = running_velocity_extraction_GAUSS(colours, names, ALL_vecs, spacer, TEST, parent_dir, typee, save, showw)

    # v = [vv, vvv]
    # return avgoflist(v)
      









###################################################################
# FUNCTION DEFINITIONS - back to flow rate!
###################################################################

# collects to correct metadata values and puts them in that nice [sag, cor, ax]
def collectmetadata(parent_dir):

    metaname = parent_dir + "/metadata.txt"
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
    a = 512-1
    b = 512-1
    c = 160-1

    sag1 = [float("%.4f" % (first[0]*0.1)), float("%.4f" % (0.1*(first[0] + (a)*space[0])))]
    cor1 = [float("%.4f" % (first[1]*0.1)), float("%.4f" % (0.1*(first[1] + (b)*space[1])))]
    ax1 = [float("%.4f" % (0.1*first[2])), float("%.4f" % (0.1*first[2]+0.1*(c)))]

    aa = [sag1, cor1, ax1]

    return aa

# converts pixel and cm coordinates
def convertpixtomm(radpix, parent_dir):
    sag, cor, ax = collectmetadata(parent_dir)

    one = (sag[1]-sag[0])/512
    two = (cor[1]-cor[0])/512
    avg = ((one + two)/2)

    return avg*radpix*10

# calculates flowrate in ml/min
# takes in radius in pixels
def flowrate(avg_vel, radpix, parent_dir):
    rad = convertpixtomm(radpix, parent_dir)
    # rad in mm AND avgvel in mm/s
    area = math.pi * rad**2
    flow = area*avg_vel #mm^3/s * (60s/1min) * (1ml/1000mm^3)
    flow1 = float("%.3f" % (flow*60/1000))
    flow2 = float("%.3f" % (flow*60/1000000))
    print("velocity: " + str(avg_vel) + " [mm/s]")
    print("average vessel radius: " + str(float("%.3f" % rad)) + " [mm]")
    print("average vessel area: " + str(float("%.3f" % area)) + " [mm^2]")
    return flow1, flow2

# calculates flowrate in ml/min
# takes in radius in mm
def flowrate2(avg_vel, rad, parent_dir):
    # rad in mm AND avgvel in mm/s
    area = math.pi * rad**2
    flow = area*avg_vel #mm^3/s * (60s/1min) * (1ml/1000mm^3)
    flow1 = float("%.3f" % (flow*60/1000000))
    flow2 = float("%.3f" % (flow*60/1000))
    print("velocity: " + str(avg_vel) + " [mm/s]")
    print("average vessel radius: " + str(float("%.3f" % rad)) + " [mm]")
    print("average vessel area: " + str(float("%.3f" % area)) + " [mm^2]")
    return flow1, flow2



























###################################################################
# MAIN
###################################################################
aaa = time.perf_counter()



# DEFINE ALL INPUT FOLDERS WE NEED TO USE FOR EXTRACTION
timebody_folder = "/Users/felicialiu/Desktop/summer2022/coding/Cardiac 1.0 CE - 3"
typee = "AORTA"
TEST = "/Users/felicialiu/Desktop/summer2022/coding/centreline_files/" + typee + ".pth"
pparent_dir = "/Users/felicialiu/Desktop/summer2022/coding"
speed = 80


# ***********
# PHASE ZERO: prep making all the folders!
# ***********

# # make the gif folder - everything will be stored here
# name = typee + "run"
# parent_dir = make_dir(pparent_dir, name)
# name = "TICPLOT_ANALYSIS"
# make_dir(parent_dir, name)
# name = "gif"
# here = make_dir(parent_dir, name)


parent_dir = pparent_dir + "/" + typee + "run"
here = parent_dir + "/gif/"




# ***********
# PHASE ONE: generating all the gifs in a "gif" folder
# ***********


# # extract all gifs into folder "gif" in "runthrough2"
# all_time = True # if True, extracted for every time interval
# how_many = 10 # capped at 151 if all_time == False
# size, spacing = size_type_chooser(typee) 
# a_plane_over_time_LOTS(parent_dir, timebody_folder, TEST, spacing, all_time, how_many, size, here, speed, typee)
# print("\n\n***\nyou need 'here' for next step:", here)







# ***********
# PHASE TWO: from the gif folder, remove all bad gifs
# ***********
# need gif folder with gifs in order to run


# removers = [3, 99]
# remove_bad_gifs(typee, removers, here)





# ***********
# PHASE THREE: extracting each gif in gif folder and getting the TIC
# ***********
# need gif folder with gifs in order to run

frame_size = frame_size_chooser(typee) # under 3 uses threshold 17 instead of 23
size, spacing = size_type_chooser(typee) 
avgrad = run_tics_for_all_gifs(parent_dir, speed, frame_size, size)





# ***********
# PHASE FOUR: extracting each TIC plot and figure out velocities
# ***********
# need individual folders named "TYPEE#slice" with "plotresults.txt" files in each

# first is save, second is show
# v = run_both_velocities(TEST, parent_dir, typee, False, True)
# v = run_both_velocities_SMALL(TEST, parent_dir, typee, False, True)
# print("\n***\naverage ALL velocities: " + str(v) + " [mm/s]\n***")







# ***********
# PHASE FIVE: extracting each TIC plot and figure out velocities
# ***********
# need average velocity value and radius

# avg_vel = 13.8673 # mm/s
# radpix = 4.6318 # pixels
# avg_vel = v
# radpix = 4.7579
# flow1, flow2 = flowrate(avg_vel, radpix, parent_dir)
# print("flowrate for", typee, "is:", flow1, "[ml/min] OR", flow2, "[L/min]")

# # avg_vel = 54.353 # mm/s
# # # rad = 14.645 # mm
# # # rad = 4.545 # mm
# # rad = 6.565 # mm
# # flow1, flow2 = flowrate2(avg_vel, rad, parent_dir)





bbb = time.perf_counter()
###################################################################
# CLEAN UP
###################################################################

# just to mark the end of the script

print("\n\nWHOLE RUNTIME:")
tictoc(aaa, bbb)
print("done")

print("\n\n#######################################################")
