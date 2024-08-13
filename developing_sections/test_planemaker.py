'''
the goal of this is i want to input:
- a simvascular model
- the set of perfusion ct images

- instead of just choosing which axial slice i want.... say 100 right now
- instead i want to choose how long down the vessel (this length constraint depends on the 
  vessel and patient itself) where the ROI (the slice) is
- then where that point is, i need to extract the normal vector 
- and with that vector find the normal plane - and all the points on the normal plane (voxels)
- and figure out all the points on a 512x512x160 that correspond (mapping)
- then put those locations maybe in a giant long list? (2d/3d array) 
  like tuples maybe of the right coordinates?
- then i can sort through all the perfusion cts... 151 of them total (case 1)
- and in each, extract the right grid of all the point on the normal plane
- save that as an array
- all those arrays go in a stacked list -> numpy array
- that's the "gif_pics" which i will use in "test_intensity3.py" to run my current TIC extraction on
'''


###################################################################
# SETUP
###################################################################

# importing relevant libraries
import numpy as np
import matplotlib as plt
from pydicom import dcmread
import matplotlib.pyplot as plt
from PIL import Image as im 
import time 


print("#######################################################\n\n")



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

# prints out a dictionary in an easier way to view
def dict_printer(dict):
    print("id : [position, tangent, rotation]")
    for key in dict:
        print(key, ":", dict[key])

# checking the time of certain functions, manually place tic-toc
def tictoc(tic, toc):
    runtime = toc-tic
    print("runtime:", runtime, "\n")
# tic = time.perf_counter()
# toc = time.perf_counter()
# tictoc(tic, toc)

# prints a huge dictionary set of dictionaries
def many_dict_printer(many_dicts):

    for dict in many_dicts:
        print("*** DICTIONARY :", dict, "********************************************************")
        dict_printer(many_dicts[dict])
        print("\n")





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



###################################################################
# MAIN
###################################################################


# the print just one dictionary
TEST = "/Users/felicialiu/Desktop/summer2022/coding/centreline_files/LIMA.pth"
dict = init_path_file(TEST)
dict_printer(dict)
# scatter3Dplot(dict)
# tangent_line(dict)
loc = 56
X, Y, Z = plot3D_pos_normal(dict, loc)
# locs = [10, 20, 30, 40 , 50, 60 ,70, 80, 90, 100, 110]
# plot3D_manyloc_pos_normal(dict, locs)





###################################################################
# CLEAN UP
###################################################################

# just to mark the end of the script
print("\n\ndone")

print("\n\n#######################################################")



