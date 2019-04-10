# from matplotlib.animation import FuncAnimation
from importlib import reload
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# disable unused import warnings for casadi
from casadi import *  # pylint: disable=W0614
import numpy as np
sys.path.append('../..')
import nilo_seminar_lib as nsl  # pylint: disable=E0401
import matplotlib.animation as animation
import sys

# reload on rerun for changes in nsl
reload(nsl)

# m=2: trapezoidal rule
# m=3: Simpson's rule
m = 3
n = 10

N = m + (n-1)*(m-1)

X = np.linspace(0, 1, N)
Y = X

B = np.zeros((N, 4))
#B[0, 0] = 1
#B[0, 1] = 1
#B[0, 2] = 0
#B[0, 3] = 0
#B[-1, 3] = 1

# double parabola
b = 2*(X-0.5)**2 + 0.5
B = np.tile(np.atleast_2d(b).T, [1,4])
B[:,0] = -B[:,0] + 1
B[:,2] = -B[:,2] + 1

# prignle
#b = 2*(X-0.5)**2
#B = np.tile(np.atleast_2d(b).T, [1,4])
#B[:,0] = -B[:,0] + 1
#B[:,2] = -B[:,2] + 1

# pringle : B + 0.5; 0,2: +1
# double parabola: B + 0; 0,2 + 1

#z = SX.sym('z', N, N)
#[sfc, z, N] = nsl.surface_composite_newton_cotes(0, 1, n, m)

# without center arch
#[sfc, sym_arg, N] = nsl.constrained_surface_composite_newton_cotes(0, 1, B, n, m)

# with center arch
[sfc, sym_arg, N] = nsl.interior_constrained_surface_composite_newton_cotes(0, 1, B, n, m)
s_f = Function('s_f', [sym_arg], [sfc])

[X, Y] = np.meshgrid(X, Y)

#Z = np.sqrt(1 - X**2 - Y**2)
#Z[np.isnan(Z)] = 0
Z = 2*np.ones((N,N))
Z[0, :] = B[:, 0]
Z[:, 0] = B[:, 1]
Z[-1, :] = B[:, 2]
Z[:, -1] = B[:, 3]

#Z[N//4, N//2] = 1
#Z[3*N//4, N//2] = 1
b = np.linspace(0,1,3*N//4 - N//4)
b = -(b - 0.5)**2 + 1
Z[N//4:3*N//4,N//2] = b

print("s_f(Z): ", s_f(Z))
print("s_f(Z) - (1 - pi/4): ", s_f(Z) - (1 - np.pi/4))

z_0 = Z
z = z_0
# z_0 = np.ones((N, N))
gamma = 1
delta = 0.1
tol = 1e-4

N = max(z_0.shape)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_zlim(0, 1)

norm_gradfz = tol + 1
# takes ages to compute as of right now; perhaps try FuncAnimation at some point
it = 0
cnt = 0
while it <= 5000 and norm_gradfz >= tol:
    sigma = gamma

    sym_gradf = gradient(sfc, sym_arg)
    func_gradf = Function('func_gradf', [sym_arg], [sym_gradf])

    fz = s_f(z)
    fz = fz.toarray()

    gradfz = func_gradf(z)
    gradfz = gradfz.toarray()

    gradfz_reshaped = gradfz.T.reshape((N**2, 1))
    gradfz_sq = np.dot(gradfz_reshaped.T, gradfz_reshaped)
    norm_gradfz = np.sqrt(gradfz_sq)

    sigma = gamma
    phi = s_f(z - sigma * gradfz)
    phi = phi.toarray()
    while phi > (fz - delta * sigma * gradfz_sq):
        sigma = sigma/2
        phi = s_f(z - sigma * gradfz)
        phi = phi.toarray()
    z = z - sigma * gradfz

    gradfz = func_gradf(z)
    gradfz = gradfz.toarray()

    gradfz_reshaped = gradfz.T.reshape((N**2, 1))
    gradfz_sq = np.dot(gradfz_reshaped.T, gradfz_reshaped)
    norm_gradfz = np.sqrt(gradfz_sq)
    fz = s_f(z)
    fz = fz.toarray()
    it = it + 1    

    if(it % 10 == 0):
        #fig = plt.figure()
        #ax = fig.gca(projection='3d')
        #ax.set_zlim(0, 1)
        #ax.plot_surface(X, Y, z, color='c')
        #plt.savefig('./plot_temp/parabola_test'+ str(cnt) +'.png')
        #plt.close(fig)
        cnt = cnt + 1
        print("it, cnt: ", it, cnt)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_zlim(0, 1)
ax.plot_surface(X, Y, z, color='c')
plt.savefig('./plot_temp/parabola_test'+ str(cnt) +'.png')
plt.close(fig)

# ani = FuncAnimation(fig, update, frames=np.arange(0,4000), init_func=init, blit=True)
#ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
#                                repeat_delay=1000)
#ani.save('sphere_test.mp4')
