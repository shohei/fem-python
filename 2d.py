import getfem as gf
import numpy as np
import pyvista as pv

b= 1.0
L= 10000.0
h = 1000.0
X = np.linspace(0.0,L,4+1)
Y = np.linspace(0.0,h,2+1)

mesh = gf.Mesh('cartesian',X,Y)
mesh.export_to_vtk('mesh.vtk','ascii')

m = pv.read('mesh.vtk')

m.plot(show_edges='True',cpos='xy')

fb1 = mesh.outer_faces_with_direction([-1.0,0.0],0.01)
fb2 = mesh.outer_faces_with_direction([1.0,0.0],0.01)
LEFT = 1
RIGHT = 2

mesh.set_region(LEFT, fb1)
mesh.set_region(RIGHT, fb2)

mfu = gf.MeshFem(mesh, 2)
elements_degree = 2
mfu.set_classical_fem(elements_degree)

im = gf.Integ("IM_PRODUCT(IM_GAUSS1D(4),IM_GAUSS1D(4))")
mim = gf.MeshIm(mesh,im)

E = 205000.0 #N/mm2
nu = 0.0
md = gf.Model('real')
md.add_fem_variable('u',mfu)
md.add_initialized_data('E',E)
md.add_initialized_data('nu',nu)
md.add_isotropic_linearized_elasticity_brick_pstress(
    mim,'u','E','nu'
)

md.add_initialized_data('H',[[1.0,0.0],[0.0,1.0]])
md.add_initialized_data('r',[0.0,0.0])
md.add_generalized_Dirichlet_condition_with_multipliers(
    mim,'u',mfu,LEFT,'r','H'
)

F = -1.0
md.add_initialized_data('F',[0,F])
md.add_source_term_brick(mim, 'u','F',RIGHT)

md.solve()

U=  md.variable('u')
mfu.export_to_vtk('mfu2d.vtk','ascii',mfu,U,'U')

m = pv.read('mfu2d.vtk')
m.plot(
    scalars="U",
    cpos="xy",
    scalar_bar_args={"title":"U(mm)"}
)







