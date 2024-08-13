


###################################################################
# SETUP
###################################################################

# importing relevant libraries
from calendar import c
from ctypes import cdll
from re import I
import time 
import numpy as np
import matplotlib.pyplot as plt
from pydicom import dcmread
import os
import math
from PIL import Image as im 

print("#######################################################\n\n")










###################################################################
# VISUALIZATION FUNCTIONS
###################################################################
# uses the printer function on all matrices in a giant array
# takes in vector of pixel arrays
def print_stack(pics):
    count = 0
    for i in pics:
        print(count)
        count += 1
        printer2(i)
    return

# prints out a dictionary in an easier way to view
def dict_printer(dict):
    print("id : [position, tangent, rotation]")
    for key in dict:
        print(key, ":", dict[key])

# prints a huge dictionary set of dictionaries
def many_dict_printer(many_dicts):

    for dict in many_dicts:
        print("*** DICTIONARY :", dict, "********************************************************")
        dict_printer(many_dicts[dict])
        print("\n")

# helper function to show DICOM
def printer2(what):
    print("showing pixels...")
    # plt.imshow(what,cmap=plt.cm.gray)
    # plt.show()
    # img_2d = what.astype(float)
    # img_2d_scaled = (np.maximum(img_2d,0) / img_2d.max()) * 255.0
    # hist(img_2d_scaled)
    
    img_2d_scaled = np.uint8(what)
    # hist(img_2d_scaled)
    plt.imshow(img_2d_scaled, cmap=plt.cm.gray)
    plt.show()
    # data = im.fromarray(img_2d_scaled)
    # data.show()
    return

# helper function to show the frequency of HU values per slice
def hist(what):
    print("showing frequency over HU values...")
    plt.hist(what.flatten(), bins=80, color='c')
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    plt.show()

# takes the images list and makes the gif and saves it to the folder
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

# plots the points of a dictionary by positional values
def plot3D(dict):

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    x = extract_vec(dict, "pos", "x")
    y = extract_vec(dict, "pos", "y")
    z = extract_vec(dict, "pos", "z")

    xs, xm, ys, ym, zs, zm = axis_choices(x,y,z)

    ax.plot3D(x, y, z, 'maroon')
    # ax.scatter3D(x,y,z)
    ax.set_xlim3d(xs, xm)
    ax.set_ylim3d(ys, ym)
    ax.set_zlim3d(zs, zm)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    ax.set_title('3D line plot')
    plt.show()

# plots many dictionaries together so we can see the lines!
def plot3D_multiple(many_dicts):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    xx = []
    yy = []
    zz = []

    for key in many_dicts:
        dict = many_dicts[key]
        x = extract_vec(dict, "pos", "x")
        y = extract_vec(dict, "pos", "y")
        z = extract_vec(dict, "pos", "z")
        ax.plot3D(x, y, z, 'maroon')
        # ax.scatter3D(x,y,z)
        xx.append(max(x))
        xx.append(min(x))
        yy.append(max(y))
        yy.append(min(y))
        zz.append(max(z))
        zz.append(min(z))

    xs, xm, ys, ym, zs, zm = axis_choices(xx,yy,zz)
    print(xs, xm, ys, ym, zs, zm)

    ax.set_title('3D line plot')
    ax.set_xlim3d(xs, xm)
    ax.set_ylim3d(ys, ym)
    ax.set_zlim3d(zs, zm)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    plt.show()

# just plots tangent
def plot3D_tan(dict):

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    x = extract_vec(dict, "tan", "x")
    y = extract_vec(dict, "tan", "y")
    z = extract_vec(dict, "tan", "z")

    xs, xm, ys, ym, zs, zm = axis_choices(x,y,z)

    ax.plot3D(x, y, z, 'maroon')
    # ax.scatter3D(x,y,z)
    ax.set_xlim3d(xs, xm)
    ax.set_ylim3d(ys, ym)
    ax.set_zlim3d(zs, zm)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    ax.set_title('3D line plot')
    plt.show()

# plots the points of a dictionary by positional values
def plot3D_pos_normal(dict, loc):

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    x = extract_vec(dict, "pos", "x")
    y = extract_vec(dict, "pos", "y")
    z = extract_vec(dict, "pos", "z")

    xs, xm, ys, ym, zs, zm = axis_choices(x,y,z)

    X, Y, Z, out = make_normal_line_plane(dict, loc)

    if out == True:
        return 0, 0, 0

    ax.plot3D(x, y, z, 'maroon')
    ax.scatter3D(x,y,z)
    ax.contour3D(X, Y, Z, 500, cmap = "binary")

    ax.set_xlim3d(xs, xm)
    ax.set_ylim3d(ys, ym)
    ax.set_zlim3d(zs, zm)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('3D line plot')

    plt.show()

    return X, Y, Z

# plots the points of a dictionary by positional values
def plot3D_manyloc_pos_normal(dict, locs):

    for i in range(len(locs)):
        print("showing location:", locs[i])
        plot3D_pos_normal(dict, locs[i])

# plots normal up and side vectors so we can visualise
def plot_vectors_plane(nor, up, side, point, locs):
    
    print("dots:", dot(nor, up), dot(nor, side), dot(side, up))

    xnor = [point[0], point[0]+nor[0]]
    ynor = [point[1], point[1]+nor[1]]
    znor = [point[2], point[2]+nor[2]]

    xside = [point[0], point[0]+side[0]]
    yside = [point[1], point[1]+side[1]]
    zside = [point[2], point[2]+side[2]]

    xup = [point[0], point[0]+up[0]]
    yup = [point[1], point[1]+up[1]]
    zup = [point[2], point[2]+up[2]]

    # plane's scatter plots
    x = []
    y = []
    z = []
    for i in range(len(locs)):
        for j in range(len(locs[i])):
            x.append(locs[i][j][0])
            y.append(locs[i][j][1])
            z.append(locs[i][j][2])

    ax = plt.axes(projection ="3d")
    ax.plot3D(xnor, ynor, znor, 'maroon')
    ax.plot3D(xside, yside, zside, 'green')
    ax.plot3D(xup, yup, zup, 'blue')
    ax.scatter3D(x, y, z, color = "red")
    
    # ax.set_xlim3d(200, 312)
    # ax.set_ylim3d(200, 312)
    # ax.set_zlim3d(200, 280)
    ax.set_xlim3d(0, 512)
    ax.set_ylim3d(0, 512)
    ax.set_zlim3d(0, 512)
    plt.title("showing normal(r) + up(b) + side(g) vector and nor plane")
    plt.show()

# plotter for the lines of newpoints so that it corresponds
def remapped_sanitycheck(pixel_grid, newpts):

    # check intensities to make sure relatively high
    # for i in newpts:
        # print(pixel_grid[i[2]][i[0]][i[1]])
    
    # see what it looks like
    x = []
    y = []
    z = []
    for i in newpts:
        x.append(i[0])
        y.append(i[1])
        z.append(i[2])
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(x, y, z, 'maroon')
    # ax.scatter3D(x,y,z)
    ax.set_xlim3d(0, len(pixel_grid[0]))
    ax.set_ylim3d(0, len(pixel_grid[0][0]))
    ax.set_zlim3d(0, len(pixel_grid))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    ax.set_title('3D line plot')
    plt.show()
  
# plots the little frame moving along the vessel
def plot_likesim(locas, newpts, pixel_grid, typee):

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
        print('hi', count)
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
    name = typee + "plots"
    namee = '/Users/felicialiu/Desktop/summer2022/coding/gifs/' + name + '.gif'
    images[0].save(namee, 
            save_all = True, 
            append_images = images[1:], 
            duration=60, 
            loop=0)

# just manually prints out what would otherwise be seen in gifs
def plot_gifsandgraphs(gif, locas, newpts, pixel_grid):

    for w in range(len(newpts)):
        intensity = gif[w]
        locs = locas[w]
        centre = newpts[w+1]
        printer2(intensity)
        # see what it looks like
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
        ax.set_xlim3d(0, len(pixel_grid[0]))
        ax.set_ylim3d(0, len(pixel_grid[0][0]))
        ax.set_zlim3d(0, len(pixel_grid))
        ax.set_xlabel('sagital')
        ax.set_ylabel('coronal')
        ax.set_zlabel('axial')
        ax.set_title(typee + "line plot with moving normal plane")
        plt.show()
    
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



###################################################################
# HELPER FUNCTION DEFINITIONS
###################################################################

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

# checking the time of certain functions, manually place tic-toc
def tictoc(tic, toc):
    runtime = toc-tic
    print("runtime:", runtime, "\n")
# tic = time.perf_counter()
# toc = time.perf_counter()
# tictoc(tic, toc)

# just so i can make 3D matrices of controllable size
def make_small_matrix_thing(x, y, z):
    whole = []

    for i in range(1, z+1):
        layer = []
        for j in range(y):
            layer.append(x * [i])
        layer = np.array(layer)
        whole.append(layer)
    whole = np.array(whole)
    return whole

# makes it's magnitude = 1
def normalise_vec(vec):
    mag = math.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)

    if mag == 0:
        return vec

    mult = 1/mag
    vec = [element * mult for element in vec]
    return vec

# function will move us around the plane using our up and left vectors
# -up is down and -side is right
def m(start, num1, up, num2, side):
    # first do all our side moves then the up/down moves
    x = start[0] + num2*side[0] + num1*up[0]
    y = start[1] + num2*side[1] + num1*up[1]
    z = start[2] + num2*side[2] + num1*up[2]
    return [x, y, z]

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

# manually does cross product
def dot(vec1, vec2):
    return vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2]

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

# returns a list from a dict of everytype of values
def extract_vec(dict, typee, vec):

    if typee == "pos":
        first = 0
    elif typee == "tan":
        first = 1
    elif typee == "rot":
        first = 2
    
    if vec == "x":
        sec = 0
    elif vec == "y":
        sec = 1
    elif vec == "z":
        sec = 2
    
    done = []
    for key in dict:
        lists = dict[key] # 3x3 matrix
        value = lists[first][sec]
        done.append(value)
    
    return done

# say x runs 5-6, but we need range 10, then 10/2 is 5, 5-6 centre is 5.5, 
# so our range runs 0.5 to 10.5
def centre_nums(vec, range):
    minn = min(vec)
    maxx = max(vec)
    centre = (maxx+minn)/2
    adder = range/2
    s = centre - adder
    m = centre + adder
    return s, m

# i want normal axis dimensions and as small as possible
# feed it x y z vectors and it finds the biggest range and chooses 
# centres for the other two to compensate
def axis_choices(x,y,z):
    range_x = max(x) - min(x)
    range_y = max(y) - min(y)
    range_z = max(z) - min(z)
    range = max(range_x, range_y, range_z) + 2
    xs, xm = centre_nums(x, range)
    ys, ym = centre_nums(y, range)
    zs, zm = centre_nums(z, range)

    return xs, xm, ys, ym, zs, zm

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
    











###################################################################
# MAIN FUNCTION DEFINITIONS
###################################################################
 
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

# extract a lot of dictionaries at the same time
def many_dictionaries(par_dir, listt):
    print("making MANY dictionaries...")
    tic = time.perf_counter()

    master_dict = {}
    for i in range(len(listt)):
        name = par_dir + listt[i] + ".pth"
        dict = init_path_file(name)
        master_dict[listt[i]] = dict
    
    toc = time.perf_counter()
    print("done MASTER dictionaries.")
    tictoc(tic, toc)
    return master_dict

# can choose anywhere from 1 to 99
def make_normal_line_plane(dict, loc):

    # sanity check to make sure location is valid
    if loc < 1:
        print(loc, "is not a valid location. must be between 1 and", len(dict)-2)
        return 0, 0, 0, True
    if loc > len(dict)-2:
        print(loc, "is not a valid location. must be between 1 and", len(dict)-2)
        return 0, 0, 0, True

    # gives us locations for the point before and after
    less_id = "point" + str(loc - 1)
    lpos = dict[less_id][0]
    more_id = "point" + str(loc + 1)
    mpos = dict[more_id][0]

    normal = [mpos[0] - lpos[0], mpos[1] - lpos[1] , mpos[2] - lpos[2]]
    point = dict["point" + str(loc)][0]

    xx = extract_vec(dict, "pos", "x")
    yy = extract_vec(dict, "pos", "y")
    zz = extract_vec(dict, "pos", "z")

    xs, xm, ys, ym, zs, zm = axis_choices(xx,yy,zz)

    x = np.linspace(xs, xm, 30)
    y = np.linspace(ys, ym, 30)
    X, Y = np.meshgrid(x, y)

    # a(x-xo) + b(y-yo) + c(z-zo) = 0
    # z = -1/c * (a(x-xo) + b(y-yo)) + zo
    a = normal[0]
    b = normal[1]
    c = normal[2]
    xo = point[0]
    yo = point[1]
    zo = point[2]
    Z = -1/c * (a*(X-xo) + b*(Y-yo)) + zo

    return X, Y, Z, False

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

# how finely do we spin the body around? that's level
# which axis
def spin_body(pixel_grid, point, size, axis, level):

    levels = np.linspace(0, 2*math.pi, level)
    normals = []
    if axis == "z":
        # normal vector has no z and goes around with xs and ys
        # actual spinning
        for i in levels:
            x = math.cos(i)
            y = math.sin(i)
            z = 0
            normals.append([x,y,z])
    if axis == "y":
        # normal vector has no y and goes around with xs, and zs
        # like doing flips over a bar
        for i in levels:
            x = math.cos(i)
            y = 0
            z = math.sin(i)
            normals.append([x,y,z])
    if axis == "x":
        # normal vector has no x and goes around with ys, and zs
        # like cartwheels
        for i in levels:
            x = 0
            y = math.cos(i)
            z = math.sin(i)
            normals.append([x,y,z])
    
    pre_gif = []

    for i in range(len(normals)):
        normal = normals[i]
        pixels = voxel_plane_extraction(pixel_grid, point, normal, size)
        pre_gif.append(pixels)
        print("finished slide", i)

    pre_gif = np.array(pre_gif)
    
    return pre_gif

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
def mini_map(linepts, pixel_grid):
    # all the ones i'm getting from simvascular's image navigator
    sag1 = [-9.19, 16.66]
    sag2 = [0, len(pixel_grid[0])-1]
    cor1 = [-10.76, 15.10]
    cor2 = [0, len(pixel_grid[0][0])-1]
    # ax1 = [210.05, 234.05]
    # ax1 = [215, 225]
    ax1 = [205.00, 220.90]
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
def position_remapper(dict, pixel_grid):
    linepts = []
    for key in dict:
        linepts.append(dict[key][0])

    newpts = mini_map(linepts, pixel_grid)
    # remapped_sanitycheck(pixel_grid, newpts)

    return newpts
  
# now to look at normal planes along these new axis things
def looking_along(pixel_grid, dict, size):
    newpts = position_remapper(dict, pixel_grid)

    gif = []
    locas = []
    # for each of those points we want to see the normal plane
    for i in range(1, len(newpts)-1):
        print("mapping location", i)
        lpos = newpts[i-1]
        point = newpts[i]
        mpos = newpts[i+1]
        normal = [mpos[0] - lpos[0], mpos[1] - lpos[1] , mpos[2] - lpos[2]]
        intgrid = voxel_plane_extraction(pixel_grid, point, normal, size)
        locs = bring_back_locs(point, normal, size)
        locas.append(locs)
        gif.append(my_colourmap(intgrid))
        # printer2(intgrid)
    
    # newpts is 2 longer on both sides so we need to crop it a little

    return gif, locas, newpts

# now a function will extract the same plane over the whole time period
# define the place along the line we want this from
# folder is used to get the body over time in ct
# TEST is the .pth folder to extract centre points
# choose lets us identify the normal plane from the first slice
# outputs that one slice over times which we can see if we want to save!
def a_plane_over_time(folder, TEST, choose, all_time, how_many, size):

    # first we prep the body by extracting the non extended cts
    # use True to extract only up till certain time
    # otherwise it will extract the whole thing
    pixelgrid4D = extract_4DCTpixels(folder, all_time, how_many)

    # also prep the pathline into a dictionary
    dict = init_path_file(TEST)

    # now we define what the vessel looks like at time zero
    newpts = position_remapper(dict, extend_3Dbody(pixelgrid4D[0]))
    
    # we can define the plane of interests points by getting the normal
    point, normal, worked = make_normal_simpler(newpts, choose)
    if worked == False:
        return [], False# the chosen index was out of range
    locs = bring_back_locs(point, normal, size)
    plot_graphwithvessel(locs, newpts)

    x = input("if bad locs look, [X] for out:  ")
    if x == "X":
        print("please try new choose location.")
        return [], False
    
    # now for each body, we're going to extract those planes
    planetime_gif = []
    for i in range(len(pixelgrid4D)):
        print("extracting slide from pixel_grid",i+1)
        pixel_grid = extend_3Dbody(pixelgrid4D[i])
        intgrid = my_colourmap(voxel_plane_extraction_withlocs(pixel_grid, point, normal, size, locs))
        planetime_gif.append(intgrid)
        # printer2(intgrid)

    return planetime_gif, True

# just runs ^^ many times from choose in a vector
def a_plane_over_time_LOTS(folder, TEST, choices, all_time, how_many, size):
    for i in choices:
        choose = i
        print("\n\n\nRUNNING CHOOSE", choose)
        gif, saveable = a_plane_over_time(folder, TEST, choose, all_time, how_many, size)
        save_the_gif(typee, "slice20frames", choose, gif, 80, saveable)








###################################################################
# MAIN
###################################################################

# # we're going to import in the pictures 
# # each vector entry is the body at a certain time -  4D!
# folder = "/Users/felicialiu/Desktop/summer2022/coding/Cardiac 1.0 CE - 3"
# typee = "AORTA"
# TEST = "/Users/felicialiu/Desktop/summer2022/coding/centreline_files/" + typee + ".pth"
# all_time = True
# how_many = 100 # capped at 151
# choose = 2
# size = 200
# gif, saveable = a_plane_over_time(folder, TEST, choose, all_time, how_many, size)
# save_the_gif(typee, "slice90frames", choose, gif, 80, saveable)

# # choices = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150 , 160]
# # a_plane_over_time_LOTS(folder, TEST, choices, all_time, how_many, size)


# folder = "/Users/felicialiu/Desktop/summer2022/coding/Cardiac 1.0 CE - 3"
# TEST = "/Users/felicialiu/Desktop/summer2022/coding/centreline_files/AORTA.pth"
# dict = init_path_file(TEST)
# all_time = True
# how_many = 2
# size = 200
# pixelgrid4D = extract_4DCTpixels(folder, all_time, how_many)
# p = pixelgrid4D[0]
# gif, locas, newpts = looking_along(p, dict, size)
# print_stack(gif)



# this part is to match up the points of a centre line with the aorta
# basically just need to convert one unit of pixels to the other
folder = "/Users/felicialiu/Desktop/summer2022/coding/Cardiac 1.0 CE - 3"
all_time = False
how_many = 3
size = 200
pixelgrid4D = extract_4DCTpixels(folder, all_time, how_many)
pixel_grid = extend_3Dbody(pixelgrid4D[0]) # here's the body 3D pixels


# pathhh = "/Users/felicialiu/Desktop/summer2022/simvascmodeling/SEGMENT 70% 0.00s Cardiac 0.5 CE - 1/IM-0001-0008-0001.dcm"
# x = dcmread(pathhh) 
# pixel_grid = x.pixel_array




# def attach2(grid1, grid2):
#     grid3 = []
#     for i in range(len(grid1)):
#         row = []
#         for j in range(len(grid1[i])):
#             row.append(grid1[i][j])
#         for j in range(len(grid2[i])):
#             row.append(grid2[i][j])
#         grid3.append(row)
#     return np.array(grid3)

# print(len(pixels), len(pixel_grid))
# newgif = []
# for i in range(0, len(pixel_grid), 1):
#     print(i)
#     this = attach2(pixels[i], pixel_grid[i])
#     this = my_colourmap(this)
#     # printer2(this)
#     newgif.append(this)
# save_the_gif("typee", "line", 34, newgif, 80)


typee = "AORTA"
TEST = "/Users/felicialiu/Desktop/summer2022/coding/centreline_files/" + typee + ".pth"
dict = init_path_file(TEST)
# let's you look through alternating graph of moving frame like sim
# and the ct frame itself
# gif holds stack of images that's now 2D arrays
# locas holds stack of arrays linking each intensity pixel to it's point
# newpts holds vec of values which should be centre point for each slides vessel
gif, locas, newpts = looking_along(pixel_grid, dict, size = 200)
save_the_gif(typee, "line", 37, gif, 80)
# plot_likesim(locas, newpts, pixel_grid, typee)
# plot_gifsandgraphs(gif, locas, newpts, pixel_grid)




# # this enables us to look at slices of the body at any given angle
# pixel_grid = extend_3Dbody(pixelgrid4D[2])
# point = [250, 250, 300]
# size = 500
# normal = [1, 1, 1]
# intgrid = voxel_plane_extraction(pixel_grid, point, normal, size)
# printer2(intgrid)
# print(intgrid)
# new = my_colourmap(intgrid)
# printer2(new)
# print(new)





# # this is for looking at the whole body spinning around an axis
# pixel_grid = extend_3Dbody(pixelgrid4D[2])
# point = [250, 250, 300]
# size = 500
# axis = "z"
# level = 10
# speed = 100
# gif = spin_body(pixel_grid, point, size, axis, level)
# save_the_gif("rot", axis, level, gif, speed)






# # the print just one dictionary
# TEST = "/Users/felicialiu/Desktop/summer2022/coding/centreline_files/AORTA.pth"
# dict = init_path_file(TEST)
# # dict_printer(dict)
# # scatter3Dplot(dict)
# # tangent_line(dict)
# loc = 158
# X, Y, Z = plot3D_pos_normal(dict, loc)
# # # # this lets you print multiple views of one dictionary different normal planes
# # # locs = [10, 20, 30, 40 , 50, 60 ,70, 80, 90, 100, 110]
# # # plot3D_manyloc_pos_normal(dict, locs)



# # to print a bunch of vessels all on the same plot
# par_dir = "/Users/felicialiu/Desktop/summer2022/coding/centreline_files/"
# list = ["AORTA", "BCA", "LCC", "LSUB"]
# many_dicts = many_dictionaries(par_dir, list)
# # many_dict_printer(many_dicts)
# plot3D_multiple(many_dicts)














# AORTA = "/Users/felicialiu/Desktop/summer2022/coding/centreline_files/Aorta_centerlines.vtp"
# ALL = "/Users/felicialiu/Desktop/summer2022/coding/centreline_files/all_results_cl.vtp"


# import vtk.IO.XML as vvtk
# import vtk
# # include "vtkXMLPolyDataReader.h"

# reader = vtk.vtkXMLPolyDataReader()
# # reader.SetFileName(AORTA)
# # reader.Update()
# # polyDataOutput = reader.GetOutput()

# # pip install











###################################################################
# CLEAN UP
###################################################################

# just to mark the end of the script
print("\n\ndone")

print("\n\n#######################################################")


