import networkx as nx
from fenics import *
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import shutil
import os

# setting path
import sys
sys.path.append('../../graphnics')

from fenics_graph import *
from utils import *
from graph_examples import *
from utils import timeit
  
parameters["form_compiler"]["cpp_optimize"] = True

@timeit
def hydraulic_network_forms_custom(G, f=Constant(0), p_bc=Constant(0)):
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
    
    mesh = G.global_mesh
    print("nb vertices = ", mesh.num_entities(0))
    print("nb lines = ", mesh.num_entities(1))

    # Flux spaces on each segment, ordered by the edge list
    submeshes = list(nx.get_edge_attributes(G, 'submesh').values())
    P3s = [FunctionSpace(msh, 'CG', 3) for msh in submeshes] # FFC Call (1)
    #P2s = [FunctionSpace(msh, 'CG', 2) for msh in submeshes] # FFC Call (1)

    # Pressure space on global mesh
    P2 = FunctionSpace(mesh, 'CG', 2) # Pressure space (on whole mesh) # FFC Call (1)
    # P1 = FunctionSpace(mesh, 'CG', 1) # Pressure space (on whole mesh) # FFC Call (1)
    
    ### Function spaces
    spaces = P3s + [P2]
    W = MixedFunctionSpace(*spaces) 

    # Trial and test functions
    vphi = TestFunctions(W)
    qp = TrialFunctions(W)

    # split out the components
    qs = qp[0:G.num_edges]
    p = qp[-1]

    vs = vphi[0:G.num_edges]
    phi = vphi[-1]
    
    ## Assemble variational formulation 

    # Initialize blocks in a and L to zero
    # (so fenics-mixed-dim does not throw an error)
    dx = Measure('dx', domain=mesh)
    a = Constant(0)*p*phi*dx
    L = Constant(0)*phi*dx

    # Using methodology from firedrake we assemble the jumps as a vector
    # and input the jumps in the matrix later
    # FFC Call (7) : 6 FFC Calls for ix=0, and one with ix=1
    vecs = [[G.jump_vector(q, ix, j) for j in G.bifurcation_ixs] for ix, q in enumerate(qs)]

    # now we can index by vecs[branch_ix][bif_ix]
    
    # Assemble edge contributions to a and L
    for i, e in enumerate(G.edges):
        
        msh = G.edges[e]['submesh']
        vf = G.edges[e]['vf']
        #res = G.edges[e]['res']
        
        dx_edge = Measure("dx", domain = msh)
        ds_edge = Measure('ds', domain=msh, subdomain_data=vf)

        # Add variational terms defined on edge
        a += qs[i]*vs[i]*dx_edge        
        a -= p*G.dds(vs[i])*dx_edge
        a += phi*G.dds(qs[i])*dx_edge

        # Add boundary condition for inflow/outflow boundary node
        L += p_bc*vs[i]*ds_edge(BOUN_IN)
        L -= p_bc*vs[i]*ds_edge(BOUN_OUT)

    return (a, L, W, mesh, vecs, G)
    # Solve
    #qp0 = mixed_dim_fenics_solve_custom(a, L, W, mesh, vecs, G)
    #return qp0

@timeit
def export_results(qp0, reponame="plots/"):
    vars = qp0.split()
    q = GlobalFlux(G, vars[0:-1])
    qi = interpolate(q, VectorFunctionSpace(G.global_mesh, 'DG', 2, G.geom_dim))
    p = vars[-1]

    for i,var in enumerate(vars):
        var.rename('q'+str(i), '0.0')
        File(reponame + "q_" + str(i) + ".pvd") << var
    File(reponame + "q_global.pvd") << qi
    File(reponame + "p.pvd") << p

    return (qi, p)

if __name__ == '__main__':
    '''
    Do time profiling for hydraulic network model implemented with fenics-mixed-dim
    Args:
        customassembly (bool): whether to assemble real spaces separately as vectors 
    customassembly should lead to speed up
    '''

    print('Clearing cache')
    os.system('dijitso clean') 
    
    path = Path("./plots_perf")
    if path.exists() and path.is_dir():
        shutil.rmtree(path)

    path.mkdir(exist_ok=True)
    for n in range(2, 5):

        with (path / 'profiling.txt').open('a') as f:
            f.write("n: " + str(n) + "\n")

        G = make_tree(n, n, n)
        mesh = G.global_mesh

        # Run with cache cleared and record times
        p_bc = Expression('x[1]', degree=1)

        # Compute forms
        (a, L, W, mesh, vecs, G) = hydraulic_network_forms_custom(G, p_bc = p_bc)
        # Assemble
        (A_, b_) = mixed_dim_fenics_assembly_custom(a, L, W, mesh, vecs, G)
        # Solve
        qp0 = mixed_dim_fenics_solve_custom(A_, b_, W, mesh, G)

        q, p = export_results(qp0, reponame="plots/")
        print("q mean = ", np.mean(q.vector().get_local()))

    t_dict = timing_dict("./plots_perf")
    timing_table("./plots_perf") # Generate timings table file

    print("compute forms time = ", t_dict["hydraulic_network_forms_custom"])
    print("assembly time = ", t_dict["mixed_dim_fenics_assembly_custom"])
    print("solving time = ", t_dict["mixed_dim_fenics_solve_custom"])
    # print("export time = ", t_dict["export_results"])
    
    # fig, ax = plt.subplots()
    # ax.plot(t_dict["n"], t_dict["mixed_dim_fenics_solve_custom"])
    # plt.show()
