from dolfin import project, IntervalMesh, Constant, Expression, FunctionSpace, Function, TestFunction, grad, inner, DOLFIN_EPS, DirichletBC, dx, solve, dot, exp

import time

import numpy as np

from scipy.linalg import svd
from numpy import linalg as LA
import math
import scipy.io as sio

# Number of elements
nel = 255

# Boundary points
xmin, xmax = 0, 1

# Spatial grid
points = np.arange(xmin, xmax + 1/nel, 1/nel)

# Polynomial order of trial/test functions
p = 1

# Create mesh and function space
X     = np.arange( xmin, xmax + (xmax - xmin) / float(nel), (xmax - xmin) / float(nel) )
mesh  = IntervalMesh( nel, xmin, xmax )

# Define function space for this mesh using Continuous Galerkin
# (Lagrange) functions of order p on each element
V = FunctionSpace( mesh, "CG", p )
FOMspace = V

# Time discretization
n             = 200
T             = 2
timestep      = T / float(n)
t0            = 0
times         = np.linspace(t0 + timestep, T, num = int( ( T - t0 ) / float(timestep)))

def FOMsolver(Re):

    # Initialization and definition of parameters and coefficients
    
    S        = np.zeros( (times.shape[0]+1, nel + 1 ) )
    A0       = exp( Re / 8 )
    Re_inv   = Constant( 1 ) / Constant( Re )

    # Define initial condition
    ic       = project(Expression("x[0] / (1 + sqrt(1 / A0_e) * exp(Re_e * x[0] * x[0] / 4))",
                       A0_e = A0,
                       Re_e = Re,
                       degree = 2), V)

    # Define variables
    u        = ic.copy( deepcopy = True )
    u_next   = Function( V )
    v        = TestFunction( V )

    # Define all the terms of the weak formulation separately
    F1 = ( ( u_next - u ) / timestep * v ) * dx
    F2 = ( inner( u_next.dx(0) * u_next, v ) ) * dx
    F3 = ( Re_inv * dot( grad( v ), grad( u_next ) ) ) * dx
    F  = F1 + F2 + F3

    # This imposes a Dirichlet condition at the point x = xmin and x = xmax
    def Dirichlet_boundary(x, on_boundary):
        return x[0] < xmin + DOLFIN_EPS or x[0] > xmax - DOLFIN_EPS

    # Enforce u = 0 at x = xmin and x = xmax
    u_boundary = Constant( 0.0 )
    bc         = DirichletBC( V, u_boundary, Dirichlet_boundary )

    S[0] = np.flip( ( u.vector().get_local() ) )[:]
    # Time loop (Newton, non linear)
    for index, t in enumerate(times):
        solve( F == 0, u_next, bc )
        u.assign(u_next)
        sol = np.flip( ( u.vector().get_local() ) )
        S[index+1] = sol[ : ]
    return X, np.array([t0]+list(times)), S


def loadData():
    import gdown
    import numpy as np
    gdown.download(id = "1sKExVcPnohi0tJZOcokx9UDAxw5FsU-H", output = "FOMdata.npz", quiet=False)
    data = np.load("FOMdata.npz")
    from IPython.display import clear_output
    clear_output()
    return data['mu'][:50], data['u'][:50]


def animate(x, t, u, speed = 4):
    import dlroms.gifs as gifs
    import matplotlib.pyplot as plt
    import numpy as np
    rnd = np.random.randint(50000)
    umin, umax = u.min(), u.max()
    def drawframe(j):
        plt.figure(figsize = (4,3))
        plt.plot(x, u[j*speed])
        plt.title("t = %.2f" % t[j*speed])
        plt.axis([-0.05, 1.05, umin*1.05 - umax*0.05, -umin*0.05 + umax*1.05])
    gifs.save(drawframe, len(u)//speed, "temp%d-gif" % rnd)
    from IPython.display import Image, display
    display(Image("temp%d-gif.gif" % rnd))
    from os import remove
    remove("temp%d-gif.gif" % rnd)
