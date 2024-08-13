###################################################################
# SETUP
###################################################################

# importing relevant libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

print("#######################################################\n\n")

###################################################################
# TRYING STUFF
###################################################################


# brings in 3D axes
ax = plt.axes(projection = "3d")

# test case plots a spiral by defining x, y as functions of z
z = np.linspace(0,30,100)
x = np.sin(z)
y = np.cos(z)
# ax.plot3D(x, y, z)
# plt.show()

# test case plots a surface?
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)

def z_function(x, y):
    return np.sin(np.sqrt(x**2 + y**2))

X, Y = np.meshgrid(x, y)
Z = z_function(X, Y)
print("X", X)
print("Y", Y)
print("Z", Z)


# ax.plot_surface(X, Y, Z)
# plt.show()




###################################################################
# CLEAN UP
###################################################################

# just to mark the end of the script
print("done")

print("\n\n#######################################################")