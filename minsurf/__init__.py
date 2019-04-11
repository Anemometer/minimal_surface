#from .nilo_seminar_lib import *

import numpy as np
import scipy.integrate
# disable unused import warnings for casadi
from casadi import *  # pylint: disable=W0614

name = "minsurf"

def surface_composite_newton_cotes(left_bd, right_bd, n=20, m=2):
    """Assemble a casadi-symbolic composite Newton-Cotes
    surface integration formula for a function z(x,y)
    assumed to be discretized on the grid 
    [left_bd, right_bd]^2 with N = m + (n-1)*(m-1)
    equally spaced points in every direction.


    Arguments:
        n {[int]} -- [number of individual interval 
        partitions of [left_bd, right_bd] into intervals
        of m points each]
        m {[int]} -- [number of points within each partial
        interval of the composite Newton-Cotes procedure]
        left_bd {[int]} -- [lower boundary of the discretization
        interval in each dimension]
        right_bd {[int]} -- [lower boundary of the discretization
        interval in each dimension]

    Returns
        sfc {[SX]} -- [casadi symbolic expression of the
        resulting Newton-Cotes surface integral approximation
        where the input argument is pased as an NxN SX 
        symbolic expression]
        z {[SX]} -- [casadi symbolic expression of the
        input grid z]
    """
    # actual number of discrete points in every dimension
    N = m + (n-1)*(m-1)
    z = SX.sym('z', N, N)

    # distance between two discretization points
    h = (right_bd - left_bd)/(N-1)

    # generate Newton-Cotes coefficients for each partial interval
    print("Generating Newton-Cotes coeffs...")
    [c, B] = scipy.integrate.newton_cotes(m-1)
    c = 1/(m-1) * c
    print("...done!\n")
    # define the approximated surface function
    # initiate with zero for iterated assembly
    sfc = SX.sym('sfc', 1)
    sfc[0] = 0

    # write Newton-Cotes coefficients into symbolic vector
    coeff = SX(c)
    c = coeff

    print("Assembling surface functional...")
    s = 0

    for k in range(0, n):
        for l in range(0, n):
            for i in range(0, m):
                for j in range(0, m):

                    ind_i = k*(m-1) + i
                    ind_j = l*(m-1) + j

                    if(ind_i == N - 1):
                        dy = (z[ind_i - 1, ind_j]-z[ind_i, ind_j])/h
                    else:
                        dy = (z[ind_i + 1, ind_j]-z[ind_i, ind_j])/h

                    if(ind_j == N - 1):
                        dx = (z[ind_i, ind_j - 1]-z[ind_i, ind_j])/h
                    else:
                        dx = (z[ind_i, ind_j + 1]-z[ind_i, ind_j])/h


                    sfc = sfc + c[i]*c[j]*sqrt(1 + dx**2 + dy**2)
                    s = s + 1

    sfc = (((right_bd - left_bd)/(n))**2) * sfc
    print("...done! Performed s = ", s, "assembly operations.\n")
    return sfc,z,N


def constrained_surface_composite_newton_cotes(left_bd, right_bd, B, n=20, m=2):
    # constrained_surface_composite_newton_cotes(left_bd, right_bd, A, b, n=20, m=2):
    """Assemble a casadi-symbolic composite Newton-Cotes
    surface integration formula for a function z(x,y)
    assumed to be discretized on the grid 
    [left_bd, right_bd]^2 with N = m + (n-1)*(m-1)
    equally spaced points in every direction.


    Arguments:
        n {[int]} -- [number of individual interval 
        partitions of [left_bd, right_bd] into intervals
        of m points each]
        m {[int]} -- [number of points within each partial
        interval of the composite Newton-Cotes procedure]
        left_bd {[int]} -- [lower boundary of the discretization
        interval in each dimension]
        right_bd {[int]} -- [lower boundary of the discretization
        interval in each dimension]

    Returns
        sfc {[SX]} -- [casadi symbolic expression of the
        resulting Newton-Cotes surface integral approximation
        where the input argument is pased as an NxN SX 
        symbolic expression]
        z {[SX]} -- [casadi symbolic expression of the
        input grid z]
    """
    # actual number of discrete points in every dimension
    N = m + (n-1)*(m-1)
    z_complete = SX.sym('z', N, N)
    """
    # restriction matrix
    # put the four corners to 0.5
    A = np.zeros((4,N**2))
    A[0,0] = 1
    A[1,N-1] = 1
    A[2, N**2 - N] = 1
    A[3, N**2 - 1] = 1

    # filter out the points of z_complete which are set 
    # to the constraint parameters
    Ind = np.eye(N**2).T[list(set(np.arange(0,N**2).tolist()) - set(np.where(A==1)[1].tolist()))]
    z_unconstrained = mtimes(Ind, reshape(z_complete,(N**2,1)))
    print("z_unconstrained.shape: ", z_unconstrained.shape)
    print("z_unconstrained[:]: ", z_unconstrained)
    b = np.zeros((4,1))
    b[0] = 0
    b[1] = 0
    b[2] = 0
    b[3] = 1
    # solve system for particular solution
    w = np.linalg.lstsq(A,b)[0]
    print("w: ", w)
    assert(np.allclose(A.dot(w), b))
    # determine kernel matrix
    [Q,R] = np.linalg.qr(A.T, mode = 'complete')
    Q = Q.T[4:,:].T
    print("Q.shape: ", Q.shape)
    # true zeros or ones may have a wrong sign which
    # creates more problems for what it's worth for large 
    # dimensions
    print("Q: \n", Q)
    #Q[Q<0] = -Q[Q<0]
    #print("Q<0: \n", Q)
      
    #z = reshape(w + mtimes(Q,z_unconstrained),(N,N))
    print("np.where(w==1): ", np.where(w==1)[0])
    z_inter = reshape(z_complete,(N**2,1))
    z_inter[np.where(A==1)[1]] = 0
    z_inter[np.where(w==1)[0][-1]] = 1
    z_inter = reshape(z_inter, (N,N))
    z = z_inter
    print("z.shape: ", z.shape)
    print("z[:,0], z[:,-1]: ", z[:,0], z[:,-1])
    """

    print("z_complete: ", z_complete)

    # this crazy thing I need to do because otherwise
    # z_complete is altered and thus not purely
    # symbolic anymore which prevents us from
    # building a function for evaluation
    z_inter = reshape(z_complete, (N**2,1))
    z_inter = z_complete
    z_inter = reshape(z_inter, (N,N))
    z_inter[0,:] = B[:,0]
    z_inter[:,0] = B[:,1]
    z_inter[-1,:] = B[:,2]
    z_inter[:,-1] = B[:,3]

    print("z_inter: ", z_inter)
    print("z_complete: ", z_complete)

    z = z_inter

    print("z: ", z)

    # distance between two discretization points
    h = (right_bd - left_bd)/(N-1)

    # generate Newton-Cotes coefficients for each partial interval
    print("Generating Newton-Cotes coeffs...")
    [c, B] = scipy.integrate.newton_cotes(m-1)
    c = 1/(m-1) * c
    print("...done!\n")
    # define the approximated surface function
    # initiate with zero for iterated assembly
    sfc = SX.sym('sfc', 1)
    sfc[0] = 0

    # write Newton-Cotes coefficients into symbolic vector
    coeff = SX(c)
    c = coeff

    print("Assembling surface functional...")
    s = 0

    for k in range(0, n):
        for l in range(0, n):
            for i in range(0, m):
                for j in range(0, m):

                    ind_i = k*(m-1) + i
                    ind_j = l*(m-1) + j

                    if(ind_i == N - 1):
                        dy = (z[ind_i - 1, ind_j]-z[ind_i, ind_j])/h
                    else:
                        dy = (z[ind_i + 1, ind_j]-z[ind_i, ind_j])/h

                    if(ind_j == N - 1):
                        dx = (z[ind_i, ind_j - 1]-z[ind_i, ind_j])/h
                    else:
                        dx = (z[ind_i, ind_j + 1]-z[ind_i, ind_j])/h

                    #print("k*(m-1) + i: ", ind_i)
                    #print("l*(m-1) + j: ", ind_j, "\n")

                    sfc = sfc + c[i]*c[j]*sqrt(1 + dx**2 + dy**2)

                    # sfc = sfc + c[i]*c[j]*sqrt(1 + ((z[k*m + i, l*m + j+1] - z[k*m + i, l * m + j])/(
                    #    1/N))**2 + ((z[k * m + i+1, l*m + j] - z[k*m + i, l*m + j])/(1/N))**2)
                    s = s + 1

    sfc = (((right_bd - left_bd)/(n))**2) * sfc
    print("...done! Performed s = ", s, "assembly operations.\n")
    return sfc,z_complete,N

def interior_constrained_surface_composite_newton_cotes(left_bd, right_bd, B, n=20, m=2):
    # constrained_surface_composite_newton_cotes(left_bd, right_bd, A, b, n=20, m=2):
    """Assemble a casadi-symbolic composite Newton-Cotes
    surface integration formula for a function z(x,y)
    assumed to be discretized on the grid 
    [left_bd, right_bd]^2 with N = m + (n-1)*(m-1)
    equally spaced points in every direction.


    Arguments:
        n {[int]} -- [number of individual interval 
        partitions of [left_bd, right_bd] into intervals
        of m points each]
        m {[int]} -- [number of points within each partial
        interval of the composite Newton-Cotes procedure]
        left_bd {[int]} -- [lower boundary of the discretization
        interval in each dimension]
        right_bd {[int]} -- [lower boundary of the discretization
        interval in each dimension]

    Returns
        sfc {[SX]} -- [casadi symbolic expression of the
        resulting Newton-Cotes surface integral approximation
        where the input argument is pased as an NxN SX 
        symbolic expression]
        z {[SX]} -- [casadi symbolic expression of the
        input grid z]
    """
    # actual number of discrete points in every dimension
    N = m + (n-1)*(m-1)
    z_complete = SX.sym('z', N, N)

    print("z_complete: ", z_complete)

    # this crazy thing I need to do because otherwise
    # z_complete is altered and thus not purely
    # symbolic anymore which prevents us from
    # building a function for evaluation
    z_inter = reshape(z_complete, (N**2,1))
    z_inter = z_complete
    z_inter = reshape(z_inter, (N,N))
    z_inter[0,:] = B[:,0]
    z_inter[:,0] = B[:,1]
    z_inter[-1,:] = B[:,2]
    z_inter[:,-1] = B[:,3]

    b = np.linspace(0,1,3*N//4 - N//4)
    b = -(b - 0.5)**2 + 1
    z_inter[N//4:3*N//4,N//2] = b

    #z_inter[N//4, N//2] = 1
    #z_inter[3*N//4, N//2] = 1

    print("z_inter: ", z_inter)
    print("z_complete: ", z_complete)

    z = z_inter

    print("z: ", z)

    # distance between two discretization points
    h = (right_bd - left_bd)/(N-1)

    # generate Newton-Cotes coefficients for each partial interval
    print("Generating Newton-Cotes coeffs...")
    [c, B] = scipy.integrate.newton_cotes(m-1)
    c = 1/(m-1) * c
    print("...done!\n")
    # define the approximated surface function
    # initiate with zero for iterated assembly
    sfc = SX.sym('sfc', 1)
    sfc[0] = 0

    # write Newton-Cotes coefficients into symbolic vector
    coeff = SX(c)
    c = coeff

    print("Assembling surface functional...")
    s = 0

    for k in range(0, n):
        for l in range(0, n):
            for i in range(0, m):
                for j in range(0, m):

                    ind_i = k*(m-1) + i
                    ind_j = l*(m-1) + j

                    if(ind_i == N - 1):
                        dy = (z[ind_i - 1, ind_j]-z[ind_i, ind_j])/h
                    else:
                        dy = (z[ind_i + 1, ind_j]-z[ind_i, ind_j])/h

                    if(ind_j == N - 1):
                        dx = (z[ind_i, ind_j - 1]-z[ind_i, ind_j])/h
                    else:
                        dx = (z[ind_i, ind_j + 1]-z[ind_i, ind_j])/h

                    #print("k*(m-1) + i: ", ind_i)
                    #print("l*(m-1) + j: ", ind_j, "\n")

                    sfc = sfc + c[i]*c[j]*sqrt(1 + dx**2 + dy**2)

                    # sfc = sfc + c[i]*c[j]*sqrt(1 + ((z[k*m + i, l*m + j+1] - z[k*m + i, l * m + j])/(
                    #    1/N))**2 + ((z[k * m + i+1, l*m + j] - z[k*m + i, l*m + j])/(1/N))**2)
                    s = s + 1

    sfc = (((right_bd - left_bd)/(n))**2) * sfc
    print("...done! Performed s = ", s, "assembly operations.\n")
    return sfc,z_complete,N