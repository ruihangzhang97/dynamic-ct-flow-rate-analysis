
###################################################################
# SETUP
###################################################################

# importing relevant libraries
from itertools import count
from re import I
import numpy as np
import matplotlib as plt
from pydicom import dcmread
import matplotlib.pyplot as plt
from PIL import Image as im 
import os
import time

# this only imports the main file path to the coding folder
parent_dir = "/Users/felicialiu/Desktop/summer2022/coding/"



print("#######################################################\n\n")


###################################################################
# FUNCTION DEFINITIONS
###################################################################
 
# helper function to show DICOM
def printer(what):
    print("showing DICOM...\n")
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

# uses the printer function on all matrices in a giant array
def print_stack(pics):
    for i in pics:
        printer(i)
    return

# takes the images list and makes the gif and saves it to the folder
def save_the_gif(typee, choose, end_range, here, images):
    name = typee + "gif_slice_" + str(choose) + "_with_" + str(end_range) + "_entries"
    namee = here + "/" + name + '.gif'
    images[0].save(namee, 
            save_all = True, 
            append_images = images[1:], 
            duration=60, 
            loop=0)

# function takes in lots of sets of dicoms, extracts a certain slice
# called "choose" and combines into one giant new matrix accross time
# takes whole folder of dicoms
def axial_timemaker(path, choose, end_range, here, save_pics = False, gif_make = False):
    print("AXIAL: (gif) extracting slice", choose, "from", end_range, "DICOMS...")
    tic = time.perf_counter()
    
    holder = []
    pre_gif = []

    hundreds = 0
    tens = 0
    ones = 1

    # for each DICOM in the whole folder, we're going to extract one slice
    for i in range(end_range):

        # just prepares the name of the file
        num_in = str(hundreds) + str(tens) + str(ones)
        name = path + "IM-0001-0" + num_in + "-0001.dcm"
        ones += 1
        if ones == 10:
            tens += 1
            ones = 0
            if tens == 10:
                hundreds += 1
                tens = 0
        
        # now we fix the colour and append the new grid to a premade list
        x = dcmread(name) 
        pixels = x.pixel_array[choose]
        # pixels = HU_corr(x.pixel_array[choose])
        holder.append(pixels)

        # CALL the saving function here to simulataneously keep all the photos
        # pixels is the actual array to save
        # name is what the png will be called
        if save_pics == True:
            saver(pixels, num_in)

        # if we want to make a gif, pixel arrays will be saved as images
        if gif_make == True:
            data = im.fromarray(pixels)
            pre_gif.append(data)
        
    # at the end, make holder an array
    holder = np.array(holder) 
    pre_gif = np.array(pre_gif) 

    if gif_make == True:
        save_the_gif("AXIAL", choose, end_range, here, pre_gif)
        
    toc = time.perf_counter()
    tictoc(tic, toc)
    print("done axial time extraction.\n")
    return holder

# inputting whole set of dynamic dicoms, gives stacks of matrices
# gif_pics that are better resolution
def axial_gifspawner(path, choose, end_range):
    print("AXIAL: making stack of slice", choose, "from", end_range, "DICOMS...")
    tic = time.perf_counter()
    holder = []

    hundreds = 0
    tens = 0
    ones = 1

    # for each DICOM in the whole folder, we're going to extract one slice
    for i in range(end_range):

        # just prepares the name of the file
        num_in = str(hundreds) + str(tens) + str(ones)
        name = path + "IM-0001-0" + num_in + "-0001.dcm"
        ones += 1
        if ones == 10:
            tens += 1
            ones = 0
            if tens == 10:
                hundreds += 1
                tens = 0
        
        # now we fix the colour and append the new grid to a premade list
        x = dcmread(name) 
        pixels = x.pixel_array[choose]
        
        # take the pixel array and quickly save and reextract as gif
        data = im.fromarray(pixels)
        namee = '/Users/felicialiu/Desktop/summer2022/coding/gifs/eh.gif'
        data.save(namee)
        wow = im.open(namee)
        hold = np.array(wow)
        os.remove(namee)

        holder.append(hold)
        
    # at the end, make holder an array
    holder = np.array(holder) 

    toc = time.perf_counter()
    tictoc(tic, toc)
    print("done axial gif extraction.\n")
    return holder

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
    tictoc(tic, toc)
    print("done flattening.\n")
    return flattened_array

# used as helper to save and pixel array as grayscale
def saver(array, use, here):
    name = here + "/" + use + '.png'
    plt.imsave(name, array, cmap=plt.cm.gray)
  
# runs a gradient function on an array
def get_grad_pixels(pixels):
    print("making a gradient...")
    tic = time.perf_counter()
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
    toc = time.perf_counter()
    tictoc(tic, toc)
    print("gradient done.\n")
    return grad_pixels

# checking the time of certain functions, manually place tic-toc
def tictoc(tic, toc):
    runtime = toc-tic
    print("runtime:", runtime)
# tic = time.perf_counter()
# toc = time.perf_counter()
# tictoc(tic, toc)

# opens iso and gets ready for intensity extraction
def prep_iso(iso_path):

    # opens and stores iso in that slices folder so we can compare to all sums
    image = im.open(iso_path).convert('L')
    iso = np.array(image)

    return iso

# takens in all gif pics and iso to make vector of intensity values
def intensity_extraction(gif_pics, iso):
    print("extracting intensity vector...")
    tic = time.perf_counter()
    vec = []
    for frame in gif_pics:
        
        sum = 0
        count = 0

        for i in range(len(frame)):
            for j in range(len(frame[i])):

                if iso[i][j] > 100:
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
def TIC_maker(TIC_vec, TYPE, choose, here, use):
    print("making TIC plot...")
    tic = time.perf_counter()
    x = []
    # first make the x vector:
    for i in range(len(TIC_vec)):
        x.append( float("%.1f" % (i * (0.1))))
    
    title = TYPE + ": Intensity Time Curve (TIC) of Slice " + str(choose) + " - Case 1 CABG Project"
    
    plt.plot(x, TIC_vec, label='TIC')
    plt.xlabel("time of perfusion acquisition [s]")
    plt.ylabel("intensity of the pixels [grayscale]")
    plt.title(title)
    plt.legend()
    plt.xlim(0, 16)
    plt.ylim(0, 255)

    name = here + "/" + use + '.png'
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

# THE WHOLE SHABANG!
def wholething(TYPE, choose, end_range):

    #### PHASE #1 : up until you need to make iso

    # start making folder with choose and end_range
    print("making directory...")
    directory = TYPE + "_slice" + str(choose) + "_(" + str(end_range) + "frames)"
    path = os.path.join(parent_dir, directory) 
    os.mkdir(path) 
    print("directory '% s' created.\n" % directory)

    # let's make a clear variable for this important folder path
    here = parent_dir + directory

    # now there's the option to make a gif to first visualise this dynamic cut
    # ask user input for yes or no
    val = input("would you like to make a gif for visualisation? : [enter 'yes' or 'no']\n")
    print("\n")

    if val == "yes":
        # then we want to run the make a gif function
        pathh = parent_dir + "Cardiac 1.0 CE - 3/"
        time_pics = axial_timemaker(pathh, choose, end_range, here, save_pics = False, gif_make = True)

    # now either way, we can extract the gif_pics
    pathh = parent_dir + "Cardiac 1.0 CE - 3/"
    gif_pics = axial_gifspawner(pathh, choose, end_range)

    # with the gif_pics next we will flatten them and store the photo
    flat = flattener(gif_pics)
    printer(flat)
    name = TYPE + "flat_slice_" + str(choose) + "_with_" + str(end_range) + "_entries"
    saver(flat, name, here)

    # then we'll also store the gradent version cause why not
    flatter = get_grad_pixels(flat)
    printer(flatter)
    name = TYPE + "grad_slice_" + str(choose) + "_with_" + str(end_range) + "_entries"
    saver(flatter, name, here)

    #### PHASE #2 : make iso

    # now if time to alert the user to make a iso.png 512x512 
    # and put it in the right folder, gotta be based off the flat pic
    print("please now make 'iso.png' using google drawings.")
    print("must be 512x512 pixels and put in: ", directory)
    vall = input("is 'iso.png' ready? : [enter 'yes' once complete]\n")
    print("")

    if vall == "yes":

        #### PHASE #3 : finish everything else up!


        # presumably iso.png is ready! which means we can initate the TIC process
        # first we extract the time intensity vector
        iso_path = here + "/iso.png"
        iso = prep_iso(iso_path)
        TIC_vec = intensity_extraction(gif_pics, iso)

        useful = TYPE + "plot_slice_" + str(choose) + "_with_" + str(end_range) + "_entries"
        xs = TIC_maker(TIC_vec, "AXIAL", choose, here, useful)
    
    return xs, TIC_vec

# still generates all the folders, just makes one giant plot at the end too!
def multiple_graph_one(TYPE, choices, end_range):

    titles_for_graph = []
    all_intensities = []
    all_time = []

    for i in range(len(choices)):
        choose = choices[i]
        the_x, the_vec = wholething(TYPE, choose, end_range)

        titles_for_graph.append("LOC" + str(i))
        all_time.append(the_x)
        all_intensities.append(the_vec)
    
    # now to make the master plot
    title = TYPE + ": All Intensity Time Curves (TICs) - Case 1 CABG Project"
    
    # sanity check on the everything generated
    print(choices)
    print(titles_for_graph)
    print(all_time)
    print(all_intensities)

    # making the ultimate plot
    for i in range(len(choices)):
        plt.plot(all_time[i], all_intensities[i], label=titles_for_graph[i])
    
    plt.xlabel("time of perfusion acquisition [s]")
    plt.ylabel("intensity of the pixels [grayscale]")
    plt.title(title)
    plt.legend()
    plt.xlim(0, 16)
    plt.ylim(0, 255)

    use = "TOTAL_TICS"
    here = parent_dir
    name = here + "/" + use + '.png'
    plt.savefig(name)
    plt.show()

    print("TOTAL plot created and saved.\n")

    return
    
# you gotta predefine all the terms and then you can just plot that part again :)
def just_graph(TYPE, choices, titles_for_graph, all_time, all_intensities, colours):
    print("just graphing...")
    # now to make the master plot
    title = TYPE + ": All Intensity Time Curves (TICs) - Case 1 CABG Project"
    
    # # sanity check on the everything generated
    # print(choices)
    # print(titles_for_graph)
    # print(all_time)
    # print(all_intensities)

    # making the ultimate plot
    for i in range(len(choices)):
        plt.plot(all_time[i], all_intensities[i], label=titles_for_graph[i], color=colours[i])
    
    plt.xlabel("time of perfusion acquisition [s]")
    plt.ylabel("intensity of the pixels [grayscale]")
    plt.title(title)
    plt.legend()
    plt.xlim(0, 16)
    plt.ylim(0, 255)
    plt.grid()


    # # uncomment only if we want to save this:
    # use = "TOTAL"
    # here = parent_dir
    # name = here + "/" + use + '.png'
    # plt.savefig(name)

    plt.show()

    print("that's it.\n")
    return


# you gotta predefine all the terms and then you can just plot that part again :)
# defining which let's you specify which of the slices to take
def just_some_graph(TYPE, choices, titles_for_graph, all_time, all_intensities, colours, which):
    print("just graphing...")
    # now to make the master plot
    title = TYPE + ": All Intensity Time Curves (TICs) - Case 1 CABG Project"
    
    # # sanity check on the everything generated
    # print(choices)
    # print(titles_for_graph)
    # print(all_time)
    # print(all_intensities)

    # making the ultimate plot
    for i in range(len(choices)):
        if choices[i] in which:
            plt.plot(all_time[i], all_intensities[i], label=titles_for_graph[i], color=colours[i])
    
    plt.xlabel("time of perfusion acquisition [s]")
    plt.ylabel("intensity of the pixels [grayscale]")
    plt.title(title)
    plt.legend()
    plt.xlim(0, 16)
    plt.ylim(0, 255)
    plt.grid()


    # # uncomment only if we want to save this:
    # use = "TOTAL"
    # here = parent_dir
    # name = here + "/" + use + '.png'
    # plt.savefig(name)

    plt.show()

    print("that's it.\n")
    return


# you gotta predefine all the terms and then you can just plot that part again :)
# defining which let's you specify which of the slices to take
# THIS ONE'S JSUT FOR ME TO REDEFINE CURVE BOUNDRIES
def just_some_graph_new_range(TYPE, choices, titles_for_graph, all_time, all_intensities, colours, which):
    print("just graphing...")
    # now to make the master plot
    title = TYPE + ": All Intensity Time Curves (TICs) - Case 1 CABG Project"
    
    # # sanity check on the everything generated
    # print(choices)
    # print(titles_for_graph)
    # print(all_time)
    # print(all_intensities)

    # making the ultimate plot
    for i in range(len(choices)):
        if choices[i] in which:
            plt.plot(all_time[i], all_intensities[i], label=titles_for_graph[i], color=colours[i])
    
    plt.xlabel("time of perfusion acquisition [s]")
    plt.ylabel("intensity of the pixels [grayscale]")
    plt.title(title)
    plt.legend()
    plt.xlim(6.5, 8.5)
    plt.ylim(210, 255)
    plt.grid()


    # # uncomment only if we want to save this:
    # use = "TOTAL"
    # here = parent_dir
    # name = here + "/" + use + '.png'
    # plt.savefig(name)

    plt.show()

    print("that's it.\n")
    return


###################################################################
# MAIN
###################################################################




# TYPE = "AXIAL"
# choose = 100          # height slice variable [0 - 159] 160 depth.
# end_range = 5      # time variable [2 - 151] number of dicoms.

# wholething(TYPE, choose, end_range)






# TYPE = "AXIAL"
# choices = [100, 80, 60, 40, 50, 70, 90, 110, 130, 150] 
# end_range = 151

# multiple_graph_one(TYPE, choices, end_range)





TYPE = "AXIAL"
choices = [100, 80, 60, 40, 50, 70, 90, 110, 130, 150]
titles_for_graph = ['LOC100', 'LOC80', 'LOC60', 'LOC40', 'LOC50', 'LOC70', 'LOC90', 'LOC110', 'LOC130', 'LOC150']
colours = ['#E81023', '#FF8D00' ,'#FFF100' ,'#B9D80C' ,'#049D48' ,'#00B294' ,'#07BCF3' ,'#00198F' , '#682179', '#ED008C']
all_time = [[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9, 12.0, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8, 12.9, 13.0, 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8, 13.9, 14.0, 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7, 14.8, 14.9, 15.0], [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9, 12.0, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8, 12.9, 13.0, 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8, 13.9, 14.0, 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7, 14.8, 14.9, 15.0], [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9, 12.0, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8, 12.9, 13.0, 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8, 13.9, 14.0, 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7, 14.8, 14.9, 15.0], [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9, 12.0, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8, 12.9, 13.0, 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8, 13.9, 14.0, 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7, 14.8, 14.9, 15.0], [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9, 12.0, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8, 12.9, 13.0, 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8, 13.9, 14.0, 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7, 14.8, 14.9, 15.0], [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9, 12.0, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8, 12.9, 13.0, 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8, 13.9, 14.0, 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7, 14.8, 14.9, 15.0], [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9, 12.0, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8, 12.9, 13.0, 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8, 13.9, 14.0, 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7, 14.8, 14.9, 15.0], [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9, 12.0, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8, 12.9, 13.0, 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8, 13.9, 14.0, 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7, 14.8, 14.9, 15.0], [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9, 12.0, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8, 12.9, 13.0, 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8, 13.9, 14.0, 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7, 14.8, 14.9, 15.0], [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9, 12.0, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8, 12.9, 13.0, 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8, 13.9, 14.0, 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7, 14.8, 14.9, 15.0]]
all_intensities = [[155, 153, 180, 160, 195, 173, 166, 177, 184, 218, 189, 217, 199, 210, 212, 215, 200, 207, 227, 219, 234, 233, 240, 226, 230, 242, 235, 235, 239, 240, 236, 229, 238, 231, 230, 232, 231, 231, 224, 229, 224, 224, 222, 226, 223, 222, 225, 234, 235, 241, 244, 242, 239, 241, 248, 242, 243, 244, 239, 238, 235, 245, 241, 243, 244, 243, 240, 229, 242, 246, 247, 245, 249, 246, 235, 238, 248, 245, 236, 244, 242, 231, 232, 242, 240, 233, 238, 236, 228, 224, 235, 235, 236, 241, 242, 232, 230, 242, 245, 245, 248, 248, 237, 234, 240, 248, 244, 244, 243, 241, 233, 232, 244, 242, 240, 237, 244, 237, 232, 237, 239, 237, 232, 242, 235, 231, 231, 233, 235, 232, 237, 231, 223, 219, 217, 221, 223, 224, 218, 209, 203, 198, 197, 205, 207, 211, 199, 196, 193, 195, 206], [177, 182, 175, 179, 189, 191, 182, 188, 178, 180, 194, 211, 210, 211, 221, 217, 226, 232, 236, 233, 236, 239, 245, 239, 240, 248, 248, 250, 248, 247, 240, 236, 243, 241, 247, 243, 240, 233, 226, 236, 233, 237, 236, 236, 233, 231, 241, 247, 247, 251, 253, 253, 253, 253, 252, 241, 239, 239, 241, 248, 247, 246, 233, 230, 233, 237, 246, 247, 248, 241, 237, 241, 249, 253, 252, 251, 252, 246, 241, 250, 251, 244, 242, 252, 248, 245, 251, 249, 237, 235, 248, 243, 243, 250, 250, 244, 244, 252, 251, 246, 250, 251, 253, 254, 254, 252, 238, 239, 238, 245, 249, 248, 247, 235, 230, 229, 239, 245, 249, 248, 240, 233, 230, 245, 248, 249, 249, 250, 244, 242, 247, 246, 239, 237, 238, 238, 241, 239, 236, 227, 224, 225, 223, 228, 222, 226, 216, 222, 219, 219, 220], [193, 193, 189, 191, 207, 211, 210, 217, 208, 209, 223, 229, 231, 228, 236, 231, 238, 241, 243, 243, 243, 245, 241, 240, 240, 242, 244, 247, 247, 243, 239, 237, 239, 242, 245, 244, 240, 237, 234, 236, 236, 237, 234, 229, 228, 227, 232, 238, 243, 246, 247, 244, 245, 250, 252, 248, 249, 248, 248, 247, 246, 247, 240, 239, 240, 241, 244, 245, 247, 242, 242, 244, 248, 251, 252, 252, 249, 247, 245, 250, 253, 251, 252, 253, 252, 250, 253, 251, 248, 246, 249, 247, 247, 251, 250, 249, 248, 252, 251, 249, 252, 252, 251, 250, 251, 251, 246, 246, 245, 246, 247, 246, 245, 241, 239, 237, 240, 241, 242, 242, 239, 237, 237, 243, 247, 250, 250, 249, 247, 245, 249, 249, 246, 246, 247, 247, 246, 243, 242, 237, 237, 235, 234, 236, 230, 233, 229, 231, 233, 230, 233], [180, 173, 184, 179, 197, 202, 207, 221, 222, 219, 213, 214, 221, 227, 233, 232, 230, 236, 236, 239, 244, 241, 242, 237, 233, 236, 236, 242, 238, 237, 231, 226, 229, 228, 232, 229, 227, 222, 220, 227, 230, 232, 233, 231, 230, 229, 237, 239, 239, 240, 240, 237, 239, 244, 246, 246, 247, 248, 247, 248, 250, 251, 248, 248, 248, 249, 249, 249, 249, 247, 245, 246, 251, 250, 246, 245, 247, 246, 245, 246, 243, 235, 234, 239, 241, 241, 239, 239, 233, 236, 242, 248, 247, 248, 247, 242, 244, 248, 249, 249, 249, 249, 246, 247, 250, 250, 250, 250, 250, 249, 249, 250, 250, 251, 249, 250, 250, 249, 248, 249, 249, 246, 247, 247, 246, 244, 244, 246, 244, 244, 242, 239, 235, 234, 236, 238, 238, 237, 236, 231, 228, 230, 232, 234, 237, 239, 235, 235, 231, 235, 236], [136, 131, 118, 142, 133, 152, 149, 154, 151, 146, 164, 153, 175, 171, 182, 185, 185, 193, 180, 209, 206, 212, 216, 214, 219, 219, 228, 237, 235, 232, 227, 228, 230, 232, 241, 239, 236, 234, 230, 233, 237, 239, 240, 235, 230, 231, 235, 240, 245, 246, 242, 239, 234, 236, 239, 240, 240, 237, 231, 228, 228, 235, 237, 239, 237, 234, 231, 232, 237, 242, 246, 246, 245, 243, 242, 246, 249, 250, 250, 248, 244, 243, 244, 248, 248, 249, 248, 246, 243, 243, 245, 247, 250, 249, 247, 245, 245, 246, 248, 249, 248, 244, 243, 240, 242, 243, 244, 242, 238, 237, 234, 235, 237, 239, 241, 238, 237, 234, 235, 238, 242, 245, 245, 244, 241, 239, 239, 242, 245, 246, 244, 241, 238, 236, 239, 241, 243, 242, 240, 236, 234, 236, 237, 241, 240, 239, 233, 234, 233, 233, 236], [132, 122, 125, 133, 140, 147, 154, 140, 148, 147, 144, 163, 172, 178, 181, 174, 176, 169, 196, 187, 219, 200, 204, 212, 201, 218, 214, 230, 225, 229, 228, 224, 234, 235, 237, 238, 234, 233, 231, 236, 238, 242, 238, 240, 228, 239, 232, 246, 246, 249, 244, 241, 239, 238, 241, 243, 244, 241, 237, 233, 235, 237, 240, 241, 240, 237, 236, 237, 241, 247, 248, 248, 246, 243, 243, 244, 248, 250, 250, 249, 245, 246, 245, 249, 250, 250, 249, 246, 245, 244, 249, 249, 251, 249, 248, 244, 248, 248, 251, 251, 250, 249, 246, 243, 244, 246, 247, 246, 242, 240, 237, 239, 240, 243, 244, 242, 241, 239, 238, 240, 244, 247, 246, 245, 243, 242, 242, 244, 247, 247, 247, 242, 239, 238, 241, 244, 244, 244, 241, 238, 238, 240, 241, 244, 242, 242, 237, 236, 235, 236, 241], [114, 111, 121, 122, 132, 134, 138, 143, 141, 141, 141, 156, 160, 171, 165, 169, 175, 171, 185, 183, 205, 200, 209, 201, 205, 214, 213, 228, 225, 223, 215, 221, 225, 227, 232, 231, 229, 222, 226, 231, 232, 236, 230, 236, 234, 234, 235, 240, 243, 245, 242, 238, 238, 238, 238, 242, 240, 238, 235, 233, 234, 236, 237, 240, 238, 237, 236, 235, 239, 242, 246, 245, 245, 245, 243, 244, 246, 249, 249, 246, 247, 243, 244, 246, 249, 249, 246, 245, 242, 242, 246, 249, 250, 250, 248, 244, 245, 247, 250, 250, 248, 247, 244, 244, 245, 245, 247, 244, 241, 239, 237, 238, 240, 241, 241, 240, 240, 240, 238, 242, 245, 247, 246, 245, 245, 241, 241, 244, 245, 246, 245, 241, 238, 236, 238, 242, 242, 242, 241, 235, 236, 234, 240, 238, 241, 240, 235, 236, 231, 237, 237], [102, 105, 99, 101, 119, 114, 122, 120, 128, 118, 123, 139, 133, 145, 142, 156, 142, 141, 168, 154, 190, 171, 201, 184, 183, 191, 189, 213, 206, 221, 207, 209, 218, 217, 221, 220, 224, 217, 221, 226, 225, 228, 230, 233, 224, 233, 238, 244, 241, 243, 243, 239, 240, 240, 242, 245, 244, 243, 239, 237, 238, 238, 243, 243, 244, 240, 242, 240, 242, 247, 249, 250, 247, 247, 244, 244, 248, 250, 251, 248, 247, 243, 243, 246, 249, 249, 247, 246, 240, 243, 245, 250, 249, 247, 247, 243, 248, 250, 252, 251, 251, 250, 246, 248, 248, 249, 247, 247, 246, 244, 243, 244, 245, 243, 245, 243, 242, 243, 241, 244, 245, 249, 249, 248, 246, 244, 243, 244, 248, 245, 244, 241, 241, 239, 240, 244, 242, 241, 241, 241, 236, 236, 240, 240, 241, 240, 240, 237, 235, 232, 241], [99, 101, 101, 106, 134, 117, 130, 125, 129, 123, 124, 149, 140, 152, 138, 161, 149, 142, 167, 155, 185, 166, 205, 172, 176, 190, 190, 199, 194, 216, 182, 199, 211, 212, 216, 213, 226, 202, 218, 224, 227, 227, 234, 235, 227, 233, 237, 242, 237, 243, 242, 228, 235, 237, 237, 237, 238, 238, 231, 231, 232, 233, 235, 238, 240, 237, 237, 234, 238, 242, 247, 248, 245, 247, 241, 244, 246, 250, 251, 247, 249, 240, 242, 244, 250, 251, 248, 247, 239, 243, 245, 252, 251, 250, 249, 242, 246, 245, 250, 248, 247, 246, 241, 242, 240, 243, 241, 240, 239, 235, 233, 232, 236, 234, 239, 238, 236, 236, 236, 240, 240, 247, 248, 246, 245, 244, 243, 246, 251, 249, 247, 242, 242, 238, 241, 247, 246, 245, 242, 242, 236, 237, 243, 243, 245, 244, 245, 241, 237, 241, 240], [109, 92, 98, 92, 123, 122, 137, 126, 124, 138, 124, 160, 188, 197, 195, 178, 155, 155, 167, 203, 228, 222, 226, 211, 216, 233, 238, 239, 238, 234, 230, 229, 238, 238, 237, 229, 225, 224, 223, 228, 230, 235, 243, 232, 233, 238, 239, 246, 246, 246, 236, 235, 238, 240, 239, 239, 237, 233, 233, 234, 234, 234, 238, 241, 241, 240, 239, 238, 240, 243, 248, 250, 248, 248, 245, 247, 248, 250, 252, 251, 251, 249, 247, 247, 248, 251, 249, 249, 246, 246, 249, 253, 253, 253, 253, 248, 248, 248, 252, 252, 251, 251, 245, 242, 241, 245, 246, 245, 244, 238, 235, 235, 238, 244, 244, 244, 242, 241, 240, 244, 251, 253, 252, 249, 246, 245, 243, 250, 251, 249, 249, 245, 246, 241, 247, 247, 247, 247, 245, 245, 241, 243, 244, 246, 245, 248, 249, 248, 244, 247, 248]]
which = [50, 150]

just_graph(TYPE, choices, titles_for_graph, all_time, all_intensities, colours)
# just_some_graph(TYPE, choices, titles_for_graph, all_time, all_intensities, colours, which)
# just_some_graph_new_range(TYPE, choices, titles_for_graph, all_time, all_intensities, colours, which)






###################################################################
# CLEAN UP
###################################################################

# just to mark the end of the script
print("\n\ndone")

print("\n\n#######################################################")
