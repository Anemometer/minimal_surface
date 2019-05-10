from importlib import reload
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# disable unused import warnings for casadi
from casadi import *  # pylint: disable=W0614
import numpy as np
import minsurf as ms  # pylint: disable=E0401

# for a more detailed description: see seminar_summary.pdf, ch. 3.5

# newton-cotes quadrature rank
# m=1: midpoint rule
# m=2: trapezoidal rule
# m=3: Simpson's rule
m = 3

# number of individual intervals to divide 
# [0,1] into
n = 10

# N is the number of individual grid points
# on which the surface integrand is 
# to be evaluated
N = m + (n-1)*(m-1)

# generate the grid points in each 
# coordinate direction
X = np.linspace(0, 1, N)
Y = X

# define the surface boundary values 
# on [x,0], [x,1], [0,y], [1,y]
B = np.zeros((N, 4))

# set some individual boundarypoints if desired
#B[0, 0] = 1
#B[0, 1] = 1
#B[0, 2] = 0
#B[0, 3] = 0
#B[-1, 3] = 1

# define a quadratic function for boundary and 
# interior constraints
b = 2*(X-0.5)**2 + 0.5

# B contains - from left to right - the four 
# boundary constraint functions:
# B[:,0] = z(x,0)
# B[:,1] = z(0,y)
# B[:,2] = z(x,1)
# B[:,3] = z(1,y)
B = np.tile(np.atleast_2d(b).T, [1,4])
# in this example: set the x-axis boundaries 
# to negative quadratic functions
B[:,0] = -B[:,0] + 1
B[:,2] = -B[:,2] + 1

# other boundary curves include:

# "the pringle"
#b = 2*(X-0.5)**2
#B = np.tile(np.atleast_2d(b).T, [1,4])
#B[:,0] = -B[:,0] + 1
#B[:,2] = -B[:,2] + 1

# double positive parabola
#b = 0.5*b - 0.25
#B = np.tile(np.atleast_2d(b).T, [1,4])


# minsurf supplies two functions which yield a casadi symbolic
# argument sym_arg representing the discretized grid function z and
# a corresponding expression sfc representing the discretized surface 
# functional S(z) (see (3.7) in seminar_summary.pdf)

# the first function does not include any constraints
# it can be used to compute an approximation of the surface 
# area of the graph of a function on [0,1]^2
#[sfc, sym_arg, N] = ms.surface_composite_newton_cotes(0, 1, n, m)

# the second function includes constraints on the boundary of [0,1]^2
# and the additional option of interior constraints in the form of
# a quadratic function similar to the boundary constraints ("an arch")
# or simply two peaks at [0.25,0.5] and [0.75,0.5]
[sfc, sym_arg, N] = ms.interior_constrained_surface_composite_newton_cotes(0, 1, B, n, m, interior="arch")

# create a casadi function object from the symbolic expressions
s_f = Function('s_f', [sym_arg], [sfc])

# prepare the initial value: a grid function on a [0,1] mesh grid
[X, Y] = np.meshgrid(X, Y)

# spherical initial surface:
#Z = np.sqrt(1 - X**2 - Y**2)
#Z[np.isnan(Z)] = 0

# constant initial surface satisfying the boundary constraints
Z = 2*np.ones((N,N))
Z[0,:] = B[:,0]
Z[:,0] = B[:,1]
Z[-1,:] = B[:,2]
Z[:,-1] = B[:,3]


# and the interior constraints:

# either the described two peaks
#Z[N//4, N//2] = 1
#Z[3*N//4, N//2] = 1

# or the mentioned interior "arch" constraint
b = np.linspace(0,1,3*N//4 - N//4)
b = -(b - 0.5)**2 + 1
Z[N//4:3*N//4,N//2] = b

# as a test: evaluate the surface area functional on

# the initial value
print("s_f(Z): ", s_f(Z))

# and the initial value minus the surface area of the 
# quarter circle outside of the spherical initial value
print("s_f(Z) - (1 - pi/4): ", s_f(Z) - (1 - np.pi/4))
# for a normal spherical initial value this gives 
# 1/8th the surface area of the unit sphere

# set the initial value z_0 to z
# and the current iterate z to z_0
z_0 = Z
z = z_0


# use a quick and dirty gradient descent method with
# an Armijo line search to minimize the surface
# functional

# set the Armijo line search and gradient descent
# parameters and tolerance
gamma = 1
delta = 0.1
tol = 1e-4

# initiate the norm of the gradient to tol+1 so
# we enter the loop
norm_gradfz = tol + 1

# normally, this loop starts the gradient descent
# procedure and generates surface plots of
# every 10th iterate to be glued together into an 
# animation using ffmpeg
# this demonstration just plots the final result

# number of iterations and count of surface plot 
# snapshots of iterates so far
it = 0
cnt = 0
while it <= 5000 and norm_gradfz >= tol:
    sigma = gamma

    # compute the gradient of the surface functional
    sym_gradf = gradient(sfc, sym_arg)
    func_gradf = Function('func_gradf', [sym_arg], [sym_gradf])

    # compute the function in z (yielding a symbolic expression)
    fz = s_f(z)
    # convert the expression into a number
    fz = fz.toarray()

    # do the same with the gradient
    gradfz = func_gradf(z)
    gradfz = gradfz.toarray()

    # reshape the gradient to apply the standard
    # euclidean norm
    gradfz_reshaped = gradfz.T.reshape((N**2, 1))
    gradfz_sq = np.dot(gradfz_reshaped.T, gradfz_reshaped)
    norm_gradfz = np.sqrt(gradfz_sq)

    # perform the line search to determine the step size
    sigma = gamma
    phi = s_f(z - sigma * gradfz)
    phi = phi.toarray()
    while phi > (fz - delta * sigma * gradfz_sq):
        sigma = sigma/2
        phi = s_f(z - sigma * gradfz)
        phi = phi.toarray()
    
    # perform the gradient step
    z = z - sigma * gradfz

    # set up the function values and the gradient
    # for the next iteration
    gradfz = func_gradf(z)
    gradfz = gradfz.toarray()

    gradfz_reshaped = gradfz.T.reshape((N**2, 1))
    gradfz_sq = np.dot(gradfz_reshaped.T, gradfz_reshaped)
    norm_gradfz = np.sqrt(gradfz_sq)
    fz = s_f(z)
    fz = fz.toarray()
    it = it + 1   

    # take a surface plot snapshot of every 10th iterate
    if(it % 10 == 0):
        #fig = plt.figure()
        #ax = fig.gca(projection='3d')
        #ax.set_zlim(0, 1)
        #ax.plot_surface(X, Y, z, color='c')
        #plt.savefig('./plot_temp/surface_test'+ str(cnt) +'.png')
        #plt.close(fig)
        cnt = cnt + 1
        print("it, norm_gradfz: ", it, norm_gradfz)

# make a surface plot of the final iterate
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_zlim(0, 1)
ax.plot_surface(X, Y, z, color='c')
plt.savefig('./plot_temp/surface_test'+ str(it) +'.png')
plt.close(fig)
