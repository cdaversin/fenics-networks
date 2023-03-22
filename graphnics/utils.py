from fenics import *
import scipy.sparse as sp
from petsc4py import PETSc
from pathlib import Path

from time import perf_counter
from functools import wraps
import pandas as pd


def timeit(func):
    """
    Prints and saves to 'profiling.txt' the execution time of func
    Args:
        func: function to time
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        end = perf_counter()
        time_info = f'{func.__name__}: {end - start:.3f} s \n'
        
        # Write to profiling file
        p = Path("./plots_perf")
        p.mkdir(exist_ok=True)
        with (p / 'profiling.txt').open('a') as f:
            f.write(time_info)

        return result

    return wrapper


def timing_dict(outdir_path: str):
    """
    Read 'profiling.txt' and create a dictionary out of it
    Args:
       str : outdir path
    """
    p = Path(outdir_path)
    timing_file = (p / 'profiling.txt').open('r')
    timing_dict: Dict[str, List[float]] = dict()

    for line in timing_file:
        split_line = line.strip().split(':')
        keyword = split_line[0]
        value = float(split_line[1].split()[0])

        if keyword in timing_dict.keys():
            timing_dict[keyword].append(value)
        else:
            timing_dict[keyword] = [value]

    return timing_dict


def timing_table(outdir_path: str, mode="lm_spaces"):
    """
    Read 'profiling.txt' and create a table data file
    Args:
       str : outdir path
    """
    t_dict = timing_dict(outdir_path)

    if mode == "lm_spaces":
        df = pd.DataFrame({
            'n': t_dict["n"],
            'forms': t_dict["hydraulic_network_forms"],
            'assembly': t_dict["mixed_dim_fenics_assembly"],
            'solve': t_dict["mixed_dim_fenics_solve"]})
        df.to_csv(outdir_path + '/timings_lm_spaces.txt', sep='\t', index=False)
        
    elif mode == "lm_jump_vectors" :
        df = pd.DataFrame({
            'n': t_dict["n"],
            'forms': t_dict["hydraulic_network_forms_custom"],
            'assembly': t_dict["mixed_dim_fenics_assembly_custom"],
            'solve': t_dict["mixed_dim_fenics_solve_custom"]})
        df.to_csv(outdir_path + '/timings_lm_jump_vectors.txt', sep='\t', index=False)


@timeit
def call_assemble_mixed_system(a, L, qp0):
    return assemble_mixed_system(a==L, qp0)

@timeit
def mixed_dim_fenics_assembly_custom(a, L, W, mesh, L_jumps, G):
        
    # Assemble the system
    qp0 = Function(W)
    system = call_assemble_mixed_system(a, L, qp0)
    
    A_list = system[0]
    rhs_blocks = system[1]

    # Assemble jump vector forms
    jump_vecs = [[assemble(L_jump) for L_jump in rowrow] for rowrow in L_jumps] 
    
    # Convert our real rows to PetSc
    jump_vecs = [[convert_vec_to_petscmatrix(row) for row in rowrow] for rowrow in jump_vecs] 
    # and get the transpose
    jump_vecs_T = [[PETScMatrix(row.mat().transpose(PETSc.Mat())) for row in rowrow] for rowrow in jump_vecs]

    zero = zero_PETScMat(1,1)
    zero_row = zero_PETScMat(1,W.sub_space(-1).dim())
    zero_col = zero_PETScMat(W.sub_space(-1).dim(),1)
    
    ### Make new A_list with lagrange multipliers added
    
    # Size of the neqw ma
    nrows_A_old = W.num_sub_spaces()
    ncols_A_old = W.num_sub_spaces()
    
    num_bifs = len(G.bifurcation_ixs)
    
    A_list_n = []
    
    # Add all the rows corresponding to q1, q2, ... q_n
    for i in range(0, nrows_A_old-1):
        A_list_n += A_list[ i*ncols_A_old : (i+1)*ncols_A_old ] # copy in old blocks
        A_list_n += jump_vecs_T[i] # add the jump vectors for edge i

    # Add the single row corresponding to the pressure
    i+=1 # pressure row
    A_list_n += A_list[ i*ncols_A_old : (i+1)*ncols_A_old ]  # copy in old blocks
    A_list_n += [zero_col]*num_bifs # no jump in pressure so we add zero_cols 
    
    # Add rows corresponding to lagrange multipliers
    for i in range(0, num_bifs): # lagrange multiplier row
        for j in range(0, G.num_edges): 
            A_list_n += [jump_vecs[j][i]]  # copy in the jump vectors
        A_list_n += [zero_row] # no jump in pressure so we add a single zero row
        A_list_n += [zero]*num_bifs # last block corresponds to (lam, xi)
        
    rhs_blocks_n = rhs_blocks + [zero_PETScVec(1)]*num_bifs

    A_ = PETScNestMatrix(A_list_n) # recombine blocks now with Lagrange multipliers
    b_ = Vector()
    A_.init_vectors(b_, rhs_blocks_n)
    A_.convert_to_aij() # Convert MATNEST to AIJ for LU solver

    return (A_, b_)

@timeit
def mixed_dim_fenics_solve_custom(A_, b_, W, mesh, G):

    num_bifs = len(G.bifurcation_ixs)
    qp0 = Function(W)
    
    sol_ = Vector(mesh.mpi_comm(), sum([P2.dim() for P2 in W.sub_spaces()]) + num_bifs)
    #sol_ = Vector(mesh.mpi_comm(), sum([P2.dim() for P2 in W.sub_spaces()])) #TEST without Lagrange mult
    solver = PETScLUSolver()
    solver.solve(A_, sol_, b_)

    # Transform sol_ into qp0 and update qp_
    dim_shift = 0
    for s in range(W.num_sub_spaces()):
        qp0.sub(s).vector().set_local(sol_.get_local()[dim_shift:dim_shift + qp0.sub(s).function_space().dim()])
        dim_shift += qp0.sub(s).function_space().dim()
        qp0.sub(s).vector().apply("insert")
    return qp0

@timeit
def mixed_dim_fenics_assembly(a, L, W, mesh):
    # Assemble the system
    qp0 = Function(W)
    system = call_assemble_mixed_system(a, L, qp0)
    
    A_list = system[0]
    rhs_blocks = system[1]

    # Solve the system
    A_ = PETScNestMatrix(A_list) # recombine blocks
    b_ = Vector()
    A_.init_vectors(b_, rhs_blocks)
    A_.convert_to_aij() # Convert MATNEST to AIJ for LU solver

    return (A_, b_)

@timeit
def mixed_dim_fenics_solve(A_, b_, W, mesh):
    
    # Assemble the system
    qp0 = Function(W)

    sol_ = Vector(mesh.mpi_comm(), sum([P2.dim() for P2 in W.sub_spaces()]))
    solver = PETScLUSolver()
    
    solver.solve(A_, sol_, b_)

    # Transform sol_ into qp0 and update qp_
    dim_shift = 0
    for s in range(W.num_sub_spaces()):
        qp0.sub(s).vector().set_local(sol_.get_local()[dim_shift:dim_shift + qp0.sub(s).function_space().dim()])
        dim_shift += qp0.sub(s).function_space().dim()
        qp0.sub(s).vector().apply("insert")
    return qp0


def convert_vec_to_petscmatrix(vec):
    '''
    Convert a fenics vector from assemble into 
    dolfin.cpp.la.PETScMatrix
    '''
    
    # Make sparse
    sparse_vec = sp.coo_matrix(vec.get_local())
    #row = sp.coo_matrix(vec)
    
    # Init PETSC matrix
    petsc_mat = PETSc.Mat().createAIJ(size=sparse_vec.shape)
    petsc_mat.setUp()
    
    # Input values from sparse_vec
    for i,j,v in zip(sparse_vec.row, sparse_vec.col, sparse_vec.data): 
        petsc_mat.setValue(i,j,v)
    
    petsc_mat.assemble()
    
    return PETScMatrix(petsc_mat)


def zero_PETScVec(n):
    '''
    Make PETScVec with n zeros 
    '''
    
    petsc_vec = PETSc.Vec().createSeq(n)
    return PETScVector(petsc_vec)


def zero_PETScMat(n,m):
    '''
    Make n,m PETScMatrix with zeros
    '''
        
    petsc_mat = PETSc.Mat().createAIJ((n,m))
    petsc_mat.setUp()
    petsc_mat.assemble()
    
    return PETScMatrix(petsc_mat)


class CharacteristicFunction(UserExpression):
    '''
    Characteristic function on edge i
    '''
    def __init__(self, G, edge, **kwargs):
        '''
        Args:
            G (nx.graph): Network graph
            edge (int): edge index
        '''
        super().__init__(**kwargs)
        self.G=G
        self.edge=edge

    def eval_cell(self, values, x, cell):
        edge = self.G.mf[cell.index]
        values[0] = (edge==self.edge)
    
    

