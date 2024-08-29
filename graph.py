import numpy as np

class graph:

    def __init__(self, node_f, sim_pars, glob_f, edge_idx, edge_f):
        
        self.node_f = node_f
        self.sim_pars = sim_pars
        self.glob_f = glob_f
        self.edge_idx = edge_idx
        self.edge_f = edge_f