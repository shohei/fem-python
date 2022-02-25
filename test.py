import numpy as np
import scipy as sp
import pyvista as pv
import getfem as gf

X = np.array([0.0,1.0])
mesh = gf.Mesh('cartesian',X)
print(X)

fb1 = mesh.outer_faces_with_direction([-1.0],0.01)
print(fb1)
fb2 = mesh.outer_faces_with_direction([1.0],0.01)
print(fb2)

print(mesh.outer_faces_in_box([-1.0],[1.0]))

LEFT = 1
RIGHT = 2
mesh.set_region(LEFT, fb1)
mesh.set_region(RIGHT, fb2)
print(mesh)

mfu = gf.MeshFem(mesh ,1)
print(mfu)

f = gf.Fem('FEM_PK(1,1)')
mfu.set_fem(f)
print(mfu)

element_degree = 1
mfu.set_classical_fem(element_degree)

im = gf.Integ('IM_GAUSS1D(1)')
mim = gf.MeshIm(mesh, im)
print(mim)

md = gf.Model('real')
md.add_fem_variable('u',mfu)

k = 2.0
md.add_initialized_data('k',[k])
md.add_generic_elliptic_brick(mim,'u','k')

md.add_initialized_data('H',[[1.0]])
md.add_initialized_data('r',[0.0])
md.add_generalized_Dirichlet_condition_with_multipliers(
    mim,'u',mfu,LEFT,'r','H'
)

F = mfu.eval('1.0')
md.add_initialized_fem_data('F',mfu,F)
md.add_source_term_brick(mim,'u','F',RIGHT)

print(md.brick_list())


print(md.solve())

U = md.variable('u')
print(U)

mfu.export_to_vtk('mfu.vtk','ascii',mfu,U,'U')

m = pv.read('mfu.vtk')
print(m)


m.plot()

