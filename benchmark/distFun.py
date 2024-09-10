import readfof
from pyspark.sql import SparkSession
import numpy as np
import matplotlib.pyplot as plt
import math
import time

# Read data
def read_cosmo_data(file_path):

    # Read Fof
    FoF = readfof.FoF_catalog(
        file_path,           # simulation directory
        2,                   # snapnum, indicating the redshift (z=1)
        long_ids = False,
        swap = False,
        SFR = False,
        read_IDs = False
        )

    return FoF


# Get masses and positions from FoF
def get_pos_mass(FoF):

    pos = FoF.GroupPos/1e06             # Halo positions in Gpc/h 
    mass_raw = FoF.GroupMass * 1e10     # Halo masses in Msun/h

    dim = pos.shape[0]
    pos_mass_matrix = np.hstack([pos, mass_raw.reshape(dim, 1)])

    return pos_mass_matrix


# To assign simulation keys to each point in each simulation
def assign_key_to_rows(key_value_pair):

    key, array = key_value_pair

    return [(key, row) for row in array]


# Function that returns the partitions bounds as a dictionary of lists of tuples, 
# each tuple being the min and max of a dimension
def sub_box_bounds(box_number,r_link): 
    sub_length=1.0/box_number # partition length
    bounds={}
    base='box'
    sub_box_counter=1
    for x in range(0,box_number):
        for y in range(0,box_number):
            for z in range(0,box_number):
                key=base+str(sub_box_counter)
                single_bounds=[]
                centre=[x,y,z] # vertex of a sub_box corresponding to min x,y,z
                for i in range(3):
                    min_bound=round(max(0,centre[i]*sub_length-0.5*r_link),2)
                    max_bound=round(min(1,centre[i]*sub_length+sub_length+0.5*r_link),2)
                    single_bounds.append((min_bound,max_bound))
                bounds[key]=single_bounds
                sub_box_counter+=1

    return bounds


# Assign each point to a box
def assign_box(point, boxes):

    position = point[1]
    x, y, z = position
    box_assign = []
    
    for box_name, ((x_min, x_max), (y_min, y_max), (z_min, z_max)) in boxes.items():
     if (x_min <= x <= x_max) and (y_min <= y <= y_max) and (z_min <= z <= z_max):
           box_assign.append((box_name, point))
    
    return box_assign


# Convert all element of an rdd into a tuple
def convert_to_tuple(data):
    return (
        data[0],
        data[1][0],
        data[1][1],
        (float(data[1][2][0]), float(data[1][2][1]), float(data[1][2][2])),  # from array to tuple
        (float(data[1][3][0]), float(data[1][3][1]), float(data[1][3][2])),  # from array to tuple
        (float(data[1][4][0]), float(data[1][4][1]), float(data[1][4][2])),  # from array to tuple
        float(data[1][5])                
    )


# Convert vectors of an rdd into a np.array (key, ( ...)) 
def convert_to_array(data):
    return (
        data[0],
        (
            data[1],
            data[2],
            np.array([float(data[3][0]), float(data[3][1]), float(data[3][2])]),  # from tuple to array
            np.array([float(data[4][0]), float(data[4][1]), float(data[4][2])]),  # from tuple to array
            np.array([float(data[5][0]), float(data[5][1]), float(data[5][2])]),  # from tuple to array
            float(data[6]) # to standard float
        )
    )


# Graph object
class graph:

    def __init__(self, node_f, pos, sim_pars, glob_f, edge_idx, edge_f):
        
        self.node_f = node_f
        self.pos = pos
        self.sim_pars = sim_pars
        self.glob_f = glob_f
        self.edge_idx = edge_idx
        self.edge_f = edge_f


# Create graph object
def create_graph(rdd):

    sim_graph = graph(
        np.array(rdd[0])[:,3],   # node_f = masses
        np.array(rdd[0])[:,0:3], # pos
        np.array(rdd[3]),        # sim_pars
        np.array(rdd[2]),        # glob_f
        np.array(rdd[1])[:,0:2], # edge_idx
        np.array(rdd[1])[:,2:5], # edge_f
    )
    return(sim_graph)


