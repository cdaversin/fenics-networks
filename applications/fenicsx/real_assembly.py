import os
import networkx as nx
import scipy.sparse as sp
from operator import add

from ufl import TrialFunction, TestFunction, dx, inner, Constant, Measure
from mpi4py import MPI
from dolfinx import fem, io, mesh
from dolfinx import cpp as _cpp
from petsc4py import PETSc

# setting path
import sys
sys.path.append('../../graphnics')

from fenicsX_graph import *
# from utils import *
# from graph_examples import *

# Marker tags for inward/outward pointing bifurcation nodes and boundary nodes
BIF_IN = 1
BIF_OUT = 2
BOUN_IN = 3
BOUN_OUT = 4

class p_bc_expr:
    def eval(self, x):
        return np.full(x.shape[1], x[0])

def hydraulic_network_with_custom_assembly(G, f=None):
    '''
    Solve hydraulic network model 
        R q + d/ds p = 0
            d/ds q = f
    on graph G, with bifurcation condition q_in = q_out
    and custom assembly of the bifurcatio condition 

    Args:
        G (fg.FenicsGraph): problem domain
        f (df.function): source term
        p_bc (df.function): neumann bc for pressure
    '''

    #mesh = G.global_mesh
    msh = G.msh
    
    if f is None:
        f = Constant(msh, 0)

    # Flux spaces on each segment, ordered by the edge list
    submeshes = list(nx.get_edge_attributes(G, 'submesh').values())
    P3s = [fem.FunctionSpace(submsh, ("Lagrange", 3)) for submsh in submeshes]

    # Pressure space on global mesh
    P2 = fem.FunctionSpace(msh, ("Lagrange", 2)) # Pressure space (on whole mesh) # FFC Call (1)

    # Dictionnary of function spaces (to be used in solve function)
    function_spaces = {}
    for space in P3s:
        function_spaces[space._cpp_object] = space
    function_spaces[P2._cpp_object] = P2

    
    # Fluxes on each branch
    qs = []
    vs = []
    for P3 in P3s:
        qs.append(TrialFunction(P3))
        vs.append(TestFunction(P3))
    # PRessure
    p = TrialFunction(P2)
    phi = TestFunction(P2)

    
    ## Assemble variational formulation 

    dx = Measure('dx', domain=msh)
    zero = fem.Function(P2)

    # Compute jump vectors to be added into the global matrix as Lagrange multipliers
    vecs = [[G.jump_vector(q, ix, j) for j in G.bifurcation_ixs] for ix, q in enumerate(qs)]
    
    # Initialize forms
    a = [[None]*(len(submeshes) + 1) for i in range(len(submeshes) + 1)]
    L = [None]*(len(submeshes) + 1)

    # Build the global entity map
    edge_imap = msh.topology.index_map(msh.topology.dim)
    num_edges = edge_imap.size_local + edge_imap.num_ghosts
    entity_maps = {}
    for i, e in enumerate(G.edges):
        submsh = G.edges[e]['submesh']
        entity_map = G.edges[e]['entity_map']
        entity_maps.update({submsh: [entity_map.index(entity)
                                     if entity in entity_map else -1
                                     for entity in range(num_edges)]})
    
    # Assemble edge contributions to a and L
    for i, e in enumerate(G.edges):
        dx_edge = Measure("dx", domain = G.edges[e]['submesh'])
        ds_edge = ufl.Measure('ds', domain=G.edges[e]['submesh'],
                              subdomain_data=G.edges[e]['vf'])


        a[i][i] = fem.form(qs[i]*vs[i]*dx_edge)
        a[-1][i] = fem.form(phi*G.dds(qs[i])*dx, entity_maps=entity_maps)
        a[i][-1] = fem.form(-p*G.dds(vs[i])*dx, entity_maps=entity_maps)

        # Boundary condition on the correct space
        p_bc_ex = p_bc_expr()
        P1_e = fem.FunctionSpace(G.edges[e]['submesh'], ("Lagrange", 1))
        p_bc = fem.Function(P1_e)
        p_bc.interpolate(p_bc_ex.eval)

        L[i] = fem.form(p_bc*vs[i]*ds_edge(BOUN_IN) - p_bc*vs[i]*ds_edge(BOUN_OUT), entity_maps=entity_maps)
        
    # Add zero to uninitialized diagonal blocks (needed by petsc)
    a[-1][-1] = fem.form(zero*p*phi*dx)
    L[-1] = fem.form(zero*phi*dx)

    return (G, a, L, function_spaces, vecs)

def mixed_dim_fenics_solve_custom(G, a, L, function_spaces, jump_vecs):

    A = fem.petsc.assemble_matrix_block(a)
    A.assemble()
    # Get  values form A to be inserted in new bigger matrix A_new
    A_size = A.getSize()
    A_values = A.getValues(range(A_size[0]), range(A_size[1]))

    b = fem.petsc.assemble_vector_block(L, a)
    b_values = b.getValues(range(b.getSize()))
    
    # Insert jumps and Lagrange multipliers
    num_bifs = len(G.bifurcation_ixs)
    A_new_size = list( map(add, A.getSize(), (num_bifs, num_bifs)) )
    A_new = PETSc.Mat().create()
    A_new.setSizes(A_new_size)
    A_new.setUp()

    b_new_size = b.getSize() + num_bifs
    b_new = PETSc.Vec().create()
    b_new.setSizes(b_new_size)
    b_new.setUp()
    
    # Insert A entries into the appropriate rows and cols of A_new
    A_new.setValuesBlocked(range(A_size[0]), range(A_size[1]), A_values)
    b_new.setValuesBlocked(range(b.getSize()), b_values)
    
    # Convert our jump vector into PETSc matrices
    jump_vecs = [[convert_vec_to_petscmatrix(b_row) for b_row in qi] for qi in jump_vecs]

    # Insert jump vectors into A_new
    for idx_block, v in enumerate(jump_vecs):
        for idx_bif, jump_vec in enumerate(v):
            print("idx_block = ", idx_block)
            print("idx_bif = ", idx_bif)

            jump_vec_T = jump_vec.copy()
            jump_vec_T.transpose()
            
            jump_vec_values = jump_vec.getValues(range(jump_vec.getSize()[0]), range(jump_vec.getSize()[1]))
            jump_vec_T_values = jump_vec_T.getValues(range(jump_vec_T.getSize()[0]), range(jump_vec_T.getSize()[1]))
            
            A_new.setValuesBlocked(idx_bif + A_size[0], range(jump_vec.getSize()[1]*idx_block, jump_vec.getSize()[1]*(idx_block + 1)), jump_vec_values)
            A_new.setValuesBlocked(range(jump_vec_T.getSize()[0]*idx_block, jump_vec_T.getSize()[0]*(idx_block + 1)), idx_bif + A_size[1], jump_vec_T_values)

    # FIXME : Do we need to insert anything in b_new ?

    # Reassembling A_new and b_new
    A_new.assemble()
    b_new.assemble()
    print("A_new = ", A_new.view())
    print("b_new = ", b_new.view())

    # Configure solver
    ksp = PETSc.KSP().create(G.msh.comm)
    ksp.setOperators(A_new)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("superlu_dist")

    # Solve
    x = A_new.createVecLeft()
    ksp.solve(b_new, x)
    print("x = ", x.array_r)
    
    # Recover solution
    fluxes = []
    start = 0
    for i, e in enumerate(G.edges):
        q_space = function_spaces[a[i][i].function_spaces[0]]
        q = fem.Function(q_space)
        offset = q_space.dofmap.index_map.size_local * q_space.dofmap.index_map_bs
        q.x.array[:offset] = x.array_r[start:start + offset]
        q.x.scatter_forward()
        start += offset
        print("q[",i,"] = ", q.x.array)
        fluxes.append(q)

    p_space = function_spaces[a[-1][-1].function_spaces[0]]
    offset = p_space.dofmap.index_map.size_local * p_space.dofmap.index_map_bs
    pressure = fem.Function(p_space)
    pressure.x.array[:(len(x.array_r) - start)] = x.array_r[start:start + offset]
    pressure.x.scatter_forward()
    print("pressure = ", pressure.x.array)
    
    return (fluxes, pressure)

def convert_vec_to_petscmatrix(vec):
    '''
    Convert a fenics vector from assemble into 
    dolfin.cpp.la.PETScMatrix
    '''
    
    # Make sparse
    sparse_vec = sp.coo_matrix(vec)
    
    # Init PETSC matrix
    petsc_mat = PETSc.Mat().createAIJ(size=sparse_vec.shape)
    petsc_mat.setUp()
    
    # Input values from sparse_vec
    for i,j,v in zip(sparse_vec.row, sparse_vec.col, sparse_vec.data): 
        petsc_mat.setValue(i,j,v)
    
    petsc_mat.assemble()
    return petsc_mat
    
    
if __name__ == '__main__':
    '''
    Do time profiling for hydraulic network model implemented with fenics-mixed-dim
    Args:
        customassembly (bool): whether to assemble real spaces separately as vectors 
    customassembly should lead to speed up
    '''

    # Clear fenics cache
    print('Clearing cache')
    os.system('dijitso clean') 
    
    # Profile against a simple line graph with n nodes    
    n = 2
    #G = make_line_graph(n)
    G = make_Y_bifurcation()
    msh = G.msh
    
    (G, a, L, fs, vecs) = hydraulic_network_with_custom_assembly(G)
    (fluxes, pressure) = mixed_dim_fenics_solve_custom(G, a, L, fs, vecs)
    
    # Write to file
    for i,q in enumerate(fluxes):
        # with io.VTXWriter(msh.comm, "flux_" + str(i) + ".bp", q) as f:
        #     f.write(0.0)
        with io.XDMFFile(msh.comm, "flux_" + str(i) + ".xdmf", "w") as file:
            file.write_mesh(q.function_space.mesh)
            file.write_function(q)

    # with io.VTXWriter(msh.comm, "pressure.bp", pressure) as f:
    #     f.write(0.0)
    with io.XDMFFile(msh.comm, "pressure.xdmf", "w") as file:
        file.write_mesh(pressure.function_space.mesh)
        file.write_function(pressure)

