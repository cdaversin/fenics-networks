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
from networkmodels import *

import argparse
  
parameters["form_compiler"]["cpp_optimize"] = True

# lm_spaces = True
# lm_jump_vectors = False

# lm_spaces = False
# lm_jump_vectors = True


@timeit
def hydraulic_network_forms(G, f=Constant(0), p_bc=Constant(0)):
    '''
    Set up spaces and solve hydraulic network model 
        R q + d/ds p = 0
            d/ds q = f
    on graph G, with bifurcation condition q_in = q_out 

    Args:
        G (fg.FenicsGraph): problem domain
        f (df.function): source term
        p_bc (df.function): neumann bc for pressure
    '''
    
    
    mesh = G.global_mesh

    # Flux spaces on each segment, ordered by the edge list
    submeshes = list(nx.get_edge_attributes(G, 'submesh').values())
    P2s = [FunctionSpace(msh, 'CG', 2) for msh in submeshes] 
    
    # Real space on each bifurcation, ordered by G.bifurcation_ixs
    LMs = [FunctionSpace(mesh, 'R', 0) for b in G.bifurcation_ixs] 

    # Pressure space on global mesh
    P1 = FunctionSpace(mesh, 'CG', 1) # Pressure space (on whole mesh)
    
    ### Function spaces
    spaces = P2s + LMs + [P1]
    W = MixedFunctionSpace(*spaces) 

    # Trial and test functions
    vphi = TestFunctions(W)
    qp = TrialFunctions(W)

    model = HydraulicNetwork(G, f, p_bc)
    
    a = model.a(qp, vphi)
    L = model.L(vphi)
    
    return (a, L, W, mesh, G)

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
    P3s = [FunctionSpace(msh, 'CG', 3) for msh in submeshes]

    # Pressure space on global mesh
    P2 = FunctionSpace(mesh, 'CG', 2) # Pressure space (on whole mesh)
    
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
    L_jumps = [[G.jump_vector(q, ix, j) for j in G.bifurcation_ixs] for ix, q in enumerate(qs)]
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

        
    return (a, L, W, mesh, L_jumps, G)

@timeit
def export_results(qp0, reponame="plots/"):
    vars = qp0.split()
    q = GlobalFlux(G, vars[0:-1])
    #qi = interpolate(q, VectorFunctionSpace(G.global_mesh, 'DG', 2, G.geom_dim))
    qi = interpolate(q, FunctionSpace(G.global_mesh, 'DG', 1))
    p = vars[-1]

    for i,var in enumerate(vars[0:-1]):
        var.rename('q'+str(i), '0.0')
        File(reponame + "q_" + str(i) + ".pvd") << var
    #     print("q[", i ,"] = ", var.vector().get_local())
    # print("p = ", p.vector().get_local())
    File(reponame + "q_global.pvd") << qi
    File(reponame + "p.pvd") << p

    return (qi, p)
    
if __name__ == '__main__':
    '''
    Do time profiling for hydraulic network model implemented with fenics-mixed-dim
    '''

    parser = argparse.ArgumentParser(description='Time profiling - hydraulic network')
    parser.add_argument('-N', metavar='N', default=2, type=int, help='nb of generations (N)')
    parser.add_argument('-lm_spaces', default=1, type=int,
                        help='Lagrange multipliers in Real spaces')
    parser.add_argument('-lm_jump_vectors', default=0, type=int,
                        help='Lagrange multipliers added manually as jump vectors')
    args = parser.parse_args()
    print("args = ", args)

    path = Path("./plots_perf")
    if path.exists() and path.is_dir() and args.N == 2:
        shutil.rmtree(path)

    path.mkdir(exist_ok=True)
    n = args.N

    print('Clearing cache')
    os.system('dijitso clean')
        
    with (path / 'profiling.txt').open('a') as f:
        f.write("n: " + str(n) + "\n")

    G = make_tree(n, n, n)
    mesh = G.global_mesh

    # Run with cache cleared and record times
    p_bc = Expression('x[1]', degree=2)
    #p_bc = Constant(1)

    if args.lm_spaces:
        print("LM_spaces")
        # Compute forms
        (a, L, W, mesh, G) = hydraulic_network_forms(G, p_bc = p_bc)
        # Assemble
        (A_, b_) = mixed_dim_fenics_assembly(a, L, W, mesh)
        # Solve
        qp0 = mixed_dim_fenics_solve(A_, b_, W, mesh)

        q, p = export_results(qp0, reponame="plots/lm_spaces/n" + str(n))

        t_dict = timing_dict("./plots_perf")
        timing_table("./plots_perf", mode="lm_spaces") # Generate timings table file

        print("n = ", t_dict["n"])
        print("compute forms time = ", t_dict["hydraulic_network_forms"])
        print("assembly time = ", t_dict["mixed_dim_fenics_assembly"])
        print("solving time = ", t_dict["mixed_dim_fenics_solve"])
        
    elif args.lm_jump_vectors:
        print("jump_vectors")
        # Compute forms
        (a, L, W, mesh, L_jumps, G) = hydraulic_network_forms_custom(G, p_bc = p_bc)

        # Assemble
        (A_, b_) = mixed_dim_fenics_assembly_custom(a, L, W, mesh, L_jumps, G)
        # Solve
        qp0 = mixed_dim_fenics_solve_custom(A_, b_, W, mesh, G)
        # qp0 should be the same !

        q, p = export_results(qp0, reponame="plots/lm_jump_vectors/n" + str(n))
        # print("q mean = ", np.mean(q.vector().get_local()))

        t_dict = timing_dict("./plots_perf")
        timing_table("./plots_perf", mode="lm_jump_vectors") # Generate timings table file

        print("n = ", t_dict["n"])
        print("compute forms time = ", t_dict["hydraulic_network_forms_custom"])
        print("assembly time = ", t_dict["mixed_dim_fenics_assembly_custom"])
        print("solving time = ", t_dict["mixed_dim_fenics_solve_custom"])
