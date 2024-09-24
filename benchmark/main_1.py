import readfof
from pyspark.sql import SparkSession
from pyspark import SparkConf
import numpy as np
import pandas as pd
import itertools as it
import matplotlib.pyplot as plt
import math
import time
import pickle

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
def sub_box_bounds(box_number, r_link): 
    sub_length = 1.0 / box_number # partition length
    bounds = {}
    base = 'box'
    sub_box_counter = 1
    for x in range(0, box_number):
        for y in range(0, box_number):
            for z in range(0, box_number):
                key = base + str(sub_box_counter)
                single_bounds = []
                centre = [x, y, z] # vertex of a sub_box corresponding to min x,y,z
                for i in range(3):
                    min_bound = round(max(0, centre[i] * sub_length - 0.5 * r_link), 2)
                    max_bound = round(min(1, centre[i] * sub_length + sub_length + 0.5 * r_link), 2)
                    single_bounds.append((min_bound, max_bound))
                bounds[key] = single_bounds
                sub_box_counter += 1

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


output_file = 'time_output_1.pkl' #output file

partitions_default = 16 # default partitions
cut_default = 0.997 # default cut

######################### output structure building ####################################

# paramter names
par_names = ("memory", "mass cut", "partitions", "cores") 

# dictionary with params name and values
parameters = {par_names[0]: ('512m','1g','2g','3g','4g'),
             par_names[1]: (0.994,0.995,0.996,0.997,0.998),
             par_names[2]: (1,4,8,16,32),
             par_names[3]: (1,2,3,4)}


# combination dictionary: dictionary with a numeric index as a key
# and a tuple of index to identify the parameters combination
combinations = {}
for i, comb in enumerate(it.combinations(tuple(range(len(par_names))),2)):
    combinations[i] = comb

# combination names
comb_names = [(par_names[combinations[i][0]], par_names[combinations[i][1]]) for i in range(6)]

# time list
times = ['creation_time']

# output dataframe, Dictionary of dictionary of dataframes (time -> parameter pairs -> heatmap)
output_times = {}
for j in range(len(times)):
    output = {}
    for i in range(6):
        indx = pd.MultiIndex.from_product([[par_names[combinations[i][0]]], parameters[(par_names[combinations[i][0]])]])
        column = pd.MultiIndex.from_product([[par_names[combinations[i][1]]], parameters[(par_names[combinations[i][1]])]])
        output[comb_names[i]] = pd.DataFrame(0, index=indx, columns=column)
    output_times[times[j]] = output


############################### parameters looping ####################################

for pairs in combinations:

    # indexes of the parameters considered
    par1 = combinations[pairs][0]
    par2 = combinations[pairs][1]

    # looping over the two parameters' values
    for i, par_val_1 in enumerate(parameters[par_names[par1]]):
        for j, par_val_2 in enumerate(parameters[par_names[par2]]):
            
            ###################### spark context ######################
            
            # setup basic configuration
            conf = SparkConf()
            conf.setMaster("spark://master:7077")
            conf.setAppName("CosmoSparkApplicationBenchmark_1")
            
            config_dict = {}

            # fill the config dictionary 
            if par1 == 0 and par2 == 3:
                config_dict["spark.executor.memory"] = par_val_1
                config_dict["spark.executor.cores"] = par_val_2
            elif par1 == 3 and par2 == 0:
                config_dict["spark.executor.memory"] = par_val_2
                config_dict["spark.executor.cores"] = par_val_1
            elif par1 == 0:
                config_dict["spark.executor.memory"] = par_val_1
            elif par2 == 0:
                config_dict["spark.executor.memory"] = par_val_2
            elif par1 == 3:
                config_dict["spark.executor.cores"] = par_val_1
            elif par2 == 3:
                config_dict["spark.executor.cores"] = par_val_2

            # Apply all key-value pairs using setAll
            conf.setAll(config_dict.items())

            spark = SparkSession.builder\
                                .config(conf=conf)\
                                .getOrCreate()

            sc = spark.sparkContext

            
            ###################### spark context ###################### 
            
            print('--------------------------------------')
            print('spark context enable')
            print(combinations[pairs]," ", par_val_1," ", par_val_2, '\n')

            # start time
            start_time = time.time()
        
            # number of simulations to be processed
            n_sims = 2000

            # path list with simulation keys
            path_list = [(i, "/mnt/cosmo_GNN/Data/" + str(i)) for i in range(n_sims)]

            # parallelize path list and read files
            if (par1 == 2):
                fof_rdd = sc.parallelize(path_list, numSlices=par_val_1)\
                            .mapValues(read_cosmo_data)
            elif (par2 == 2):
                fof_rdd = sc.parallelize(path_list, numSlices=par_val_2)\
                            .mapValues(read_cosmo_data)
            else:
                fof_rdd = sc.parallelize(path_list, numSlices=partitions_default)\
                            .mapValues(read_cosmo_data)
                
            # get positions and masses for each point
            pos_mass_rdd = fof_rdd.mapValues(get_pos_mass)\
                                  .flatMap(assign_key_to_rows)
            # cut percentile
            if (par1 == 1):
                cut = par_val_1
            elif (par2 == 1):
                cut = par_val_2
            else:
                cut = cut_default

            # get mass cuts 
            mass_cut_rdd = fof_rdd.mapValues(get_pos_mass)\
                                  .mapValues(lambda x: np.quantile(x[:, -1], cut))

            mass_cuts = mass_cut_rdd.values().collect()

            mass_cuts = np.array(mass_cuts)

            # filter by mass
            pos_mass_rdd_filtered = pos_mass_rdd.filter(lambda x: x[1][-1] >= mass_cuts[x[0]])

            # count by key to trigger
            count_halos = pos_mass_rdd_filtered.countByKey()

            # end time
            end_time = time.time()

            # total phase 1 time
            time1 = end_time - start_time

            # putting measured time in the right place in the heatmap
            output_times[times[0]][comb_names[pairs]].iloc[i,j] = time1 

            print('\ntime for everything: ', time1, '\n')

        
            sc.stop()
            spark.stop()

with open(output_file, "wb") as fill:
    pickle.dump([par_names,parameters,combinations,comb_names,times,output_times],fill)
