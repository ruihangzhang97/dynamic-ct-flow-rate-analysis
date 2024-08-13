'''
take in those TICs for a folder and find the actual velocities!
- want velocity at everypoint along a vessel
- we know the length from that chart that tells us the length
  length comes from the title for the folder
- time is based on the shift which we'll calculate
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

# simply plots a vector on y with arbitry x
def plainplot(vec):
    print("running plain plotting...")
    
    x = []
    # first make the x vector:
    for i in range(len(vec)):
        x.append(i)
    
    title = "PLAIN PLOT"
    
    plt.plot(x, vec)
    plt.xlabel("arbitrary x")
    plt.ylabel("the given y")
    plt.yticks(np.arange(min(vec), max(vec)+1, 5.0))
    plt.title(title)
    plt.show()
    return






###################################################################
# FUNCTION DEFINITIONS - 
###################################################################

# takes in parent folder and figures out the correct folder order
# gives those folder names as a list
def folder_opener(typee, parent_dir):
    names = os.listdir(parent_dir)
    hold = []
    for name in names:
        if typee in name:
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
def plot_ALL_tics(typee, ALL_vecs, names, here, use="", save=False):
    print("graphing ALL TICs ...")
    title = typee + ": All Intensity Time Curves (TICs) - Case 1 CABG Project"

        
    # making the ultimate plot
    for i in range(len(ALL_vecs)):
        x = []
        # first make the x vector:
        for j in range(len(ALL_vecs[i])):
            x.append( float("%.1f" % (j * (0.1))))
        plt.plot(x, ALL_vecs[i], label=names[i])

    plt.xlabel("time of perfusion acquisition [s]")
    plt.ylabel("intensity of the pixels [grayscale]")
    plt.title(title)
    plt.legend()
    # plt.xlim(0, 16)
    plt.ylim(0, 255)
    plt.grid()

    if save == True:
        name = here + "/" + use + '.png'
        plt.savefig(name)

    plt.show()

    print("that's it.\n")
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

# figures out the vessel length based on the slice number
# need real length in
def name_to_length_converter(rlength, names, typee):
    
    # this will give us the points length
    # dict = init_path_file(TEST)
    # plength = len(dict)
    plength = 113

    lengths = []
    for name in names:
        num = name.replace(typee, "")
        num = num.replace("slice","")
        num = int(num)
        lengths.append(float("%.2f" % (rlength*(num/plength))))
    return lengths

# finds error between all terms
def calculate_error(one, two):
    error = 0
    for i in range(len(one)):
        error = error + abs(one[i]-two[i])**2
    return error

# reports the time to which the second delays the first
def stalker(first, second):
    times = []
    errors = []
    for i in range(round(len(first)*3/4)):
        times.append(float("%.1f" % (i * (0.1))))
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
def vecs_to_times_converter(ALL_vecs):
    times = [0.0]
    for i in range(1, len(ALL_vecs)):
        times.append(stalker(ALL_vecs[0], ALL_vecs[i]))
    return times

# just to make sure it actually looks better?
# correct all the ALL_vecs
def sanity_check_on_times(times, ALL_vecs):
    new = []
    for i in range(len(times)):
        cut = int(times[i]*10)
        ahh = ALL_vecs[i][cut:]
        # print(ahh, len(ahh))
        # plot_two("typee", ahh, ALL_vecs[i])
        new.append(ahh)
    return new

# make some plots?
def plot_lengths_times_velocities(lengths, times, here, save=False):

    # plot of x:lengths and y:times
    plt.scatter(lengths, times, label="time delay")
    plt.xlabel("lengths over the vessel with earliest frame as proximal [mm]")
    plt.ylabel("time delay of the bolus with 0mm as 0s [s]")
    plt.title("Time Delays Over the Length of the Vessel")
    plt.legend()
    plt.xlim(0,)
    plt.ylim(0,)
    plt.grid()
    if save == True:
        name = here + "/length_time.png"
        plt.savefig(name)
        # while plotting, let's save the data to a txt file too!
        name1 = "lengths over the vessel with earliest frame as proximal [mm]"
        thing1 = lengths
        name2 = "time delay of the bolus with 0mm as 0s [s]"
        thing2 = times
        name = here + "/length_time.txt"
        textfile = open(name, "w")
        textfile.write(name1 + "\n")
        for element in thing1:
            textfile.write(str(element) + "\n")
        textfile.write("\n")
        textfile.write(name2 + "\n")
        for element in thing2:
            textfile.write(str(element) + "\n")
        textfile.close()
    plt.show()

    

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
            avglengths2.append(float("%.2f" % ((alen+blen)/2)))
            vels2.append(float("%.2f" % ((blen-alen)/use)))
        else: 
            use = (btime-atime)
            avglengths.append(float("%.2f" % ((alen+blen)/2)))
            vels.append(float("%.2f" % ((blen-alen)/use)))
        avglengths3.append(float("%.2f" % ((alen+blen)/2)))
        vels3.append(float("%.2f" % ((blen-alen)/use)))
        
    # plot of x:lengths and y:velocities 
    plt.scatter(avglengths, vels, label="velocities")
    plt.scatter(avglengths2, vels2, label="bad velocities?")
    plt.xlabel("lengths over the vessel with earliest frame as proximal [mm]")
    plt.ylabel("velocity [s]")
    plt.title("Average Cross-Sectional Velocity Over the Length of the Vessel")
    plt.legend()
    plt.xlim(0,)
    plt.ylim(0,)
    plt.grid()
    if save == True:
        name = here + "/length_velocity.png"
        plt.savefig(name)
        # while plotting, let's save the data to a txt file too!
        name1 = "lengths over the vessel with earliest frame as proximal [mm]"
        thing1 = avglengths3
        name2 = "velocity [s]"
        thing2 = vels3
        name = here + "/length_velocity.txt"
        textfile = open(name, "w")
        textfile.write(name1 + "\n")
        for element in thing1:
            textfile.write(str(element) + "\n")
        textfile.write("\n")
        textfile.write(name2 + "\n")
        for element in thing2:
            textfile.write(str(element) + "\n")
        textfile.close()
    plt.show()

    return

# full velocity calculations
def running_velocity_extraction(parent_dir, typee, how_much, rlength, save=False):

    here = parent_dir

    # extracting all TICs
    ALL_vecs, names = preparing_TICS(typee, parent_dir)
    plot_ALL_tics(typee, ALL_vecs, names, here, "ALL_TICS", save)

    # smoothing the TICs to be less aggressive
    ALL_vecs = smooth_all(ALL_vecs, how_much)
    plot_ALL_tics(typee, ALL_vecs, names, here, "SMOOTH_TICS", save)
    
    # makes a vector for lengths
    lengths = name_to_length_converter(rlength, names, typee)

    # makes a vectors for times
    times = vecs_to_times_converter(ALL_vecs)
    new = sanity_check_on_times(times, ALL_vecs)
    plot_ALL_tics(typee, new, names, here, "SHIFTED_TICS", save)

    plot_lengths_times_velocities(lengths, times, here, save)

    return




###################################################################
# MAIN
###################################################################
aaa = time.perf_counter()

parent_dir = "/Users/felicialiu/Desktop/summer2022/coding/runthrough2"
typee = "AORTA"
how_much = 2 # how many of each side smoothing factor
rlength = 250 # length of vessel in mm

running_velocity_extraction(parent_dir, typee, how_much, rlength, True)

bbb = time.perf_counter()
###################################################################
# CLEAN UP
###################################################################

# just to mark the end of the script

print("\n\nWHOLE RUNTIME:")
tictoc(aaa, bbb)
print("done")

print("\n\n#######################################################")