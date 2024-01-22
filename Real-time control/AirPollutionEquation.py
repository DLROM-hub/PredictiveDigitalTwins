def FOMsolver(d1, d2, d3, d4, theta):
  from IPython.display import clear_output as clc
  from fenics import RectangleMesh, Point, FunctionSpace, TestFunction, Function, Expression, Constant, DirichletBC, exp, interpolate, assemble, dx, inner, grad, solve
  import matplotlib.pyplot as plt
  import numpy as np

  mesh = RectangleMesh(Point(-5,-4), Point(5, 4), 200, 200, diagonal = "crossed")

  V = FunctionSpace(mesh, 'CG', 1)
  v = TestFunction(V)
  u = Function(V)
  w1, w2, w3, w4 = Function(V), Function(V), Function(V), Function(V)

  I1 = Expression('(pow(x[0] + 2, 2) + pow(x[1] + 2.25, 2)) < pow(0.1, 2) ? 1 : 0', degree = 1)
  I2 = Expression('(pow(x[0] + 2, 2) + pow(x[1] + 0.75, 2)) < pow(0.1, 2) ? 1 : 0', degree = 1)
  I3 = Expression('(pow(x[0] + 2, 2) + pow(x[1] - 0.75, 2)) < pow(0.1, 2) ? 1 : 0', degree = 1)
  I4 = Expression('(pow(x[0] + 2, 2) + pow(x[1] - 2.25, 2)) < pow(0.1, 2) ? 1 : 0', degree = 1)

  B = Constant((np.cos(theta), np.sin(theta)))
  x0 = Constant(-2.0)
  y0 = Constant(0.0)
  rad = Expression('sqrt((x[0] - x0) * (x[0] - x0) + 0.0 * (x[1] - y0) * (x[1] - y0))', x0 = x0, y0 = y0, degree = 1)
  sigma_z = Expression('0.04 * rad * pow(1 + 0.0002 * rad, -0.5)', rad = rad, degree = 1)
  sigma_xy = Expression('0.12 * rad * pow(1 + 0.00025 * rad, -0.5)', rad = rad, degree = 1)
  mu = Expression('sigma_xy * sigma_xy * sqrt(pow(B[0], 2) + pow(B[1], 2))/(DOLFIN_EPS + 2 * rad)', sigma_xy = sigma_xy, rad = rad, B = B, degree = 1)

  target = Expression('(pow(x[0] - 3, 2) + pow(x[1] - 0.75, 2)) < pow(0.95, 2) ? 1 : 0', degree = 1)

  edge = 'near(x[0], -5)'
  bc_d = DirichletBC(V, Constant(0.0), edge)

  w1, w2, w3, w4 = tuple([interpolate(Constant(d), V) for d in [d1, d2, d3, d4]])

  F = inner(mu * grad(u), grad(v)) * dx + inner(B, grad(u)) * v * dx - (w1 * v * I1 * dx + w2 * v * I2 * dx + w3 * v * I3 * dx + w4 * v * I4 * dx)
  solve(F == 0, u, bc_d)
  clc()
  u = u.vector()[:]
  return mesh, u*(u>0)