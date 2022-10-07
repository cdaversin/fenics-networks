import networkx as nx
import numpy as np
#from fenics import *
import ufl
from mpi4py import MPI
from dolfinx import fem, io, mesh
from dolfinx import cpp as _cpp

import gmsh

# from utils import timeit

'''
The Graphnics class constructs fenics meshes from networkx directed graphs.

TODO: Add possibility for specifying Dirichlet bc nodes
'''


# Marker tags for inward/outward pointing bifurcation nodes and boundary nodes
BIF_IN = 1
BIF_OUT = 2
BOUN_IN = 3
BOUN_OUT = 4

class FenicsxGraph(nx.DiGraph):
    '''
    Make fenics mesh from networkx directed graph

    Attributes:
        global_mesh (df.mesh): mesh for the entire graph
        edges[i].mesh (df.mesh): submesh for edge i
        global_tangent (df.function): tangent vector for the global mesh, points along edge
        dds (function): derivative d/ds along edge
        mf (df.function): 1d meshfunction that maps cell->edge number
        vf (df.function): 0d meshfunction on  edges[i].mesh that stores bifurcation and boundary point data
        num_edges (int): number of edges in graph
    '''


    def __init__(self):
        nx.DiGraph.__init__(self)
        self.global_mesh = None # global mesh for all the edges in the graph

        gmsh.initialize()
        # Choose if Gmsh output is verbose
        gmsh.option.setNumber("General.Terminal", 0)

        
    def make_mesh(self, n=1, store_mesh=True, use_markers=False):
        '''
        Makes a fenics mesh on the graph with 2^n cells on each edge
        
        Args:
            n (int): number of refinements
            store (bool): whether to store the mesh as a class variable
        Returns:
            mesh (df.mesh): the global mesh
        
        If store=True, the full mesh is stored in self.global_mesh and a submesh is stored
        for each edge
        
        '''
        self.comm = MPI.COMM_WORLD

        self.geom_dim = len(self.nodes[1]['pos'])
        self.num_edges = len(self.edges)
        print("num edges = ", self.num_edges)

        domain = ufl.Mesh(ufl.VectorElement("Lagrange", "interval", 1))
        vertex_coords = np.asarray( [self.nodes[v]['pos'] for v in self.nodes()  ] )
        cells_array = np.asarray( [ [u, v] for u,v in self.edges() ] )

        lcar = 1/3. #FIXME : To be added as a parameter
        model = gmsh.model()
        pts = []
        lines = []
        for i,v in enumerate(vertex_coords):
            pts.append(gmsh.model.geo.addPoint(v[0], v[1], v[2], lcar))

        for i,c in enumerate(cells_array):
            lines.append(gmsh.model.geo.addLine(pts[c[0]], pts[c[1]]))
            gmsh.model.addPhysicalGroup(1, [lines[-1]], i)

        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(1)
        
        self.msh, self.subdomains, self.boundaries = io.gmshio.model_to_mesh(
            gmsh.model, comm=MPI.COMM_WORLD, rank=0, gdim=3)
        gmsh.finalize()

        with io.XDMFFile(self.comm, "mesh.xdmf", "w") as file:
            file.write_mesh(self.msh)

        tdim = self.msh.topology.dim
        gdim = self.msh.geometry.dim
        DG0 = fem.VectorFunctionSpace(self.msh, ("DG", 0), dim=gdim)
        self.global_tangent = fem.Function(DG0)

        for i, (u,v) in enumerate(self.edges):
            #edge_subdomain = self.subdomains.indices[self.subdomains.values == i]
            edge_subdomain = self.subdomains.find(i)
            self.edges[u,v]['submesh'], self.edges[u,v]['entity_map'] = mesh.create_submesh(self.msh, tdim, edge_subdomain)[0:2]
            self.edges[u,v]['tag'] = i

            # Compute tangent
            tangent = np.asarray(self.nodes[v]['pos'])-np.asarray(self.nodes[u]['pos'])
            tangent *= 1/np.linalg.norm(tangent)
            self.edges[u,v]['tangent'] = tangent
            for cell in edge_subdomain:
                self.global_tangent.x.array[gdim*cell:gdim*(cell+1)] = tangent

            self.edges[u,v]["entities"] = []
            self.edges[u,v]["b_values"] = []
                
        # Export global tangent
        self.global_tangent.x.scatter_forward()       
        with io.XDMFFile(self.comm, "tangent.xdmf", "w") as file:
            file.write_mesh(self.msh)
            file.write_function(self.global_tangent)

        # Marking the bifurcations (in/out) and boundaries (in/out) for extermities of each edges
        for n, v in enumerate(self.nodes()):
            num_conn_edges = len(self.in_edges(v)) + len(self.out_edges(v))
            if num_conn_edges == 0:
                print(f'Node {v} in G is lonely (i.e. unconnected)')

            bifurcation = bool(num_conn_edges > 1)
            boundary = bool(num_conn_edges == 1)

            for i,e in enumerate(self.in_edges(v)):
                e_msh = self.edges[e]['submesh']
                entities = mesh.locate_entities(e_msh,0,
                                                lambda x: np.logical_and(np.isclose(x[0], self.nodes[v]['pos'][0]),
                                                                         np.isclose(x[1], self.nodes[v]['pos'][1]),
                                                                         np.isclose(x[2], self.nodes[v]['pos'][2])))
                self.edges[e]["entities"].append(entities)
                if bifurcation : 
                    b_values_in = np.full(entities.shape, BIF_IN, np.intc)
                elif boundary:
                    b_values_in = np.full(entities.shape, BOUN_OUT, np.intc)
                self.edges[e]["b_values"].append(b_values_in)
                
            for i,e in enumerate(self.out_edges(v)):
                e_msh = self.edges[e]['submesh']
                entities  = mesh.locate_entities(e_msh,0,
                                                 lambda x: np.logical_and(np.isclose(x[0], self.nodes[v]['pos'][0]),
                                                                          np.isclose(x[1], self.nodes[v]['pos'][1]),
                                                                          np.isclose(x[2], self.nodes[v]['pos'][2])))
                self.edges[e]["entities"].append(entities)
                if bifurcation : 
                    b_values_out = np.full(entities.shape, BIF_OUT, np.intc)
                elif boundary:
                    b_values_out = np.full(entities.shape, BOUN_IN, np.intc)
                self.edges[e]["b_values"].append(b_values_out)

        # Creating the edges meshtags
        for i, e in enumerate(self.edges):
            e_msh = self.edges[e]['submesh']
            indices, pos = np.unique(np.hstack(self.edges[e]["entities"]), return_index=True)
            self.edges[e]['vf'] = mesh.meshtags(e_msh, 0, indices, np.hstack(self.edges[e]["b_values"])[pos])
            e_msh.topology.create_connectivity(0, 1)

            with io.XDMFFile(self.comm, "edge_" + str(i) + ".xdmf", "w") as file:
                file.write_mesh(e_msh)
                file.write_meshtags(self.edges[e]['vf'])

            point_imap = e_msh.topology.index_map(0)
            num_points = point_imap.size_local + point_imap.num_ghosts
            entity_maps = {e_msh: [self.edges[e]['entity_map'].index(entity)
                                   if entity in self.edges[e]['entity_map'] else -1
                                   for entity in range(num_points)]}

            # Create measure for integration
            #point_integration_entities = {1: []}
            point_integration_entities = {}
            cell_to_point = e_msh.topology.connectivity(1, 0)
            point_to_cell = e_msh.topology.connectivity(0, 1)

            for key in np.hstack(self.edges[e]["b_values"]):
                point_integration_entities[key] = []
                
                for pt in self.edges[e]["entities"]:
                    if pt < point_imap.size_local and pt in self.edges[e]['vf'].find(key):
                        # Get a cell connected to the point
                        cell = point_to_cell.links(pt)[0]
                        local_pt = cell_to_point.links(cell).tolist().index(pt)
                        print("edge ", i, " local_pt = ", local_pt, " marked by ", key)
                        # FIXME : Check the cell index here
                        point_integration_entities[key].extend([cell, local_pt])

            print("point_integration_entities = ", point_integration_entities)
            self.edges[e]['ds'] = ufl.Measure("ds", subdomain_data=point_integration_entities, domain=e_msh)

def test_fenics_graph():
    # Make simple y-bifurcation

    G = FenicsxGraph()

    # Example with on ebifurcation
    G.add_nodes_from([0, 1, 2, 3])
    G.nodes[0]['pos']=[0,0,0]
    G.nodes[1]['pos']=[0,0.5,0]
    G.nodes[2]['pos']=[-0.5,1,0]
    G.nodes[3]['pos']=[0.5,1,0]

    G.add_edge(0,1)
    G.add_edge(1,2)
    G.add_edge(1,3)

    G.make_mesh()
    
    mesh = G.global_mesh       

if __name__ == '__main__':
    test_fenics_graph()
