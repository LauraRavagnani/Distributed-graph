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



output_file = 'time_output_2.pkl' #output file

partitions_default = 16 #default partitions
cut_default = 0.997 #default cut

######################### output structure building ####################################

# paramter names
par_names = ("memory", "partitions", "cores") 

# dictionary with params name and values
parameters = {par_names[0]: ('512m','1g','2g','3g','4g'),
             par_names[1]: (1,4,8,16,32),
             par_names[2]: (1,2,3,4)}

# combination dictionary: dictionary with a numeric index as a key
# and a tuple of index to identify the parameters combination
combinations ={}
for i,comb in enumerate(it.combinations(tuple(range(len(par_names))),2)):
    combinations[i]=comb

# combination names
comb_names = [(par_names[combinations[i][0]],par_names[combinations[i][1]]) for i in range(3)]

# time list
times = ['clustering_time','graphing_time']

# output dataframe,  Dictionary of dictionary of dataframes (time -> parameter pairs -> heatmap)
output_times = {}
for j in range(len(times)):
    output = {}
    for i in range(3):
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
    for i,par_val_1 in enumerate(parameters[par_names[par1]]):
        for j,par_val_2 in enumerate(parameters[par_names[par2]]):
            
            ###################### spark context ######################
            
            # setup basic configuration
            conf = SparkConf()
            conf.setMaster("spark://master:7077")
            conf.setAppName("CosmoSparkApplicationBenchmark_2")
            
            config_dict = {}

            # fill the config dictionary 
            if par1 == 0 and par2 == 2:
                config_dict["spark.executor.memory"] = par_val_1
                config_dict["spark.executor.cores"] = par_val_2
            elif par1 == 2 and par2 == 0:
                config_dict["spark.executor.memory"] = par_val_2
                config_dict["spark.executor.cores"] = par_val_1
            elif par1 == 0:
                config_dict["spark.executor.memory"] = par_val_1
            elif par2 == 0:
                config_dict["spark.executor.memory"] = par_val_2
            elif par1 == 2:
                config_dict["spark.executor.cores"] = par_val_1
            elif par2 == 2:
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


            # number of simulations to be processed
            n_sims = 900

            # path list with simulation keys
            path_list = [(i, "/mnt/cosmo_GNN/Data/" + str(i)) for i in range(n_sims)]

            # parallelize path list and read files
            if (par1 == 1):
                fof_rdd = sc.parallelize(path_list, numSlices=par_val_1)\
                            .mapValues(read_cosmo_data)
            elif (par2 == 1):
                fof_rdd = sc.parallelize(path_list, numSlices=par_val_2)\
                            .mapValues(read_cosmo_data)
            else:
                fof_rdd = sc.parallelize(path_list, numSlices=partitions_default)\
                            .mapValues(read_cosmo_data)
                
            # get positions and masses for each point
            pos_mass_rdd = fof_rdd.mapValues(get_pos_mass)\
                              .flatMap(assign_key_to_rows)
            # cut percentile
            cut = cut_default

            # get mass cuts 
            mass_cut_rdd = fof_rdd.mapValues(get_pos_mass)\
                                  .mapValues(lambda x: np.quantile(x[:, -1], cut))
        
            mass_cuts = mass_cut_rdd.values().collect()

            mass_cuts = np.array(mass_cuts)

            # filter by mass
            pos_mass_rdd_filtered = pos_mass_rdd.filter(lambda x: x[1][-1] >= mass_cuts[x[0]])

            # number of halos in each simulation
            n_halos = pos_mass_rdd_filtered.countByKey()

            # clustering star time
            start_time = time.time()

            boxes = sub_box_bounds(4, 0.2)

            # masses rdd ---> (simkey, mass)
            mass_rdd = pos_mass_rdd_filtered.mapValues(lambda x: x[3])

            # positions rdd ---> (simkey, pos)
            pos_rdd = pos_mass_rdd_filtered.mapValues(lambda x: x[:3])

            # indexed positions rdd (point indexes)
            # --> (simkey, (point_idx, array(x, y, z)) )
            idx_pos_rdd = pos_rdd.groupByKey()\
                                 .flatMapValues(lambda vals: enumerate(vals))

            # indexed positions rdd with box assigned
            # --> ( simkey_boxkey, (point_idx, array(x, y, z)) )
            idx_pos_box_rdd = idx_pos_rdd.flatMapValues(lambda p: assign_box(p, boxes))\
                                         .map(lambda x: (str(x[0]) + '_' + x[1][0], x[1][1]))

            # obtain all the possible point pairs for each simulation clustered by boxes
            # --> ( (simkey_boxkey, (idx, array)), (simkey_boxkey, (idx, array)) )
            cartesian_rdd = idx_pos_box_rdd.groupByKey()\
                                           .flatMapValues(lambda points: [(p1,p2) for p1 in points for p2 in points])\
                                           .map(lambda x: ((x[0], x[1][0]),(x[0], x[1][1])))

            # compute differences between every pair 
            # --> (simkey_boxkey, (idx1, idx2, coord1, coord2, diff_coord))
            diff_rdd = cartesian_rdd.map(lambda x:(x[0][0],(x[0][1][0], x[1][1][0], x[0][1][1],  x[1][1][1] , x[0][1][1]-x[1][1][1])))

            # --> (simkey_boxkey, (idx1, idx2, coord1, coord2, diff_coord, norm))
            pairs_dist_rdd_with_box = diff_rdd.mapValues(lambda x: (x[0], x[1], x[2], x[3], x[4], np.linalg.norm(x[4])))

            pairs_dist_rdd_no_box = pairs_dist_rdd_with_box.map(lambda x: (int(x[0].split('_')[0]), (x[1])))\
                                                           .map(convert_to_tuple)\
                                                           .distinct()\
                                                           .map(convert_to_array)
        
            linked_pairs_dist_rdd = pairs_dist_rdd_no_box.filter(lambda x: x[1][-1] <= 0.2)

            # count by key to trigger
            pair_count = linked_pairs_dist_rdd.countByKey()

            # end clustering time
            time2 = time.time() - start_time

            print('clustering time:', time2)

            # save clustering time
            output_times[times[0]][comb_names[pairs]].iloc[i,j] = time2 

            # graph start time
            graph_start_time = time.time()

            # pairs rdd
            pairs_rdd = linked_pairs_dist_rdd.mapValues(lambda x: (x[0], x[1]))

            # centroids positions
            halo_centroids = pos_rdd.reduceByKey(lambda x,y: (x+y)/2)

            # joined rdd with halo centroids positions
            joined_rdd = linked_pairs_dist_rdd.join(halo_centroids)

            # distance between each point from each pair and halo centroid
            row_col_diff_rdd = joined_rdd.mapValues(
                lambda x: (
                    x[0][0],        # idx_i
                    x[0][1],        # idx_j
                    x[0][2] - x[1], # row
                    x[0][3] - x[1], # col
                    x[0][4],        # diff
                    x[0][5]         # dist
                    ))

            # normalizing 
            normalized_rdd = row_col_diff_rdd.mapValues(
                lambda x: (
                    x[0],                      # idx_i
                    x[1],                      # idx_j
                    x[2]/np.linalg.norm(x[2]), # row_normalized
                    x[3]/np.linalg.norm(x[3]), # col_normalized
                    x[4]/np.linalg.norm(x[4]), # s_ij
                    x[5]/0.2                   # |d_ij|/r 
                )
            )

            # edge attributes
            edge_attr_rdd = normalized_rdd.mapValues(
                lambda x: (
                    x[0],
                    x[1],
                    np.dot( x[2].T, x[3] ), # cos(alpha)
                    np.dot( x[2].T, x[4] ), # cos(beta)
                    x[5]                    # |d_ij|/r 
                )
            )

            # group by simulation
            grouped_idx_pos_rdd = pos_mass_rdd_filtered.groupByKey()\
                                                       .mapValues(list)

            grouped_edge_rdd = edge_attr_rdd.groupByKey()\
                                            .mapValues(list)

            # parallelize simulation parameters file and global features
            sim_pars_file = np.loadtxt("/mnt/cosmo_GNN/latin_hypercube_params.txt", dtype=float)
            param_rdd = sc.parallelize([(i, el) for i, el in enumerate(sim_pars_file)])

            u = sc.parallelize([(i[0], math.log10(i[1])) for i in n_halos.items()])

            # graph rdd (a graph for each simulation)
            # masses, positions, simulation parameters, global features, edge indexes, edge features
            raw_graph_rdd = grouped_idx_pos_rdd.join(grouped_edge_rdd)\
                                               .join(u)\
                                               .join(param_rdd)\
                                               .mapValues(lambda x: (x[0][0][0], x[0][0][1], x[0][1], x[1]))

            graph_rdd = raw_graph_rdd.mapValues(lambda x: create_graph(x))   

            tot_graphs = graph_rdd.count()

            time3 = time.time() - graph_start_time

            print('graph time:', time3)
        
            output_times[times[1]][comb_names[pairs]].iloc[i,j] = time3 
        
            sc.stop()
            spark.stop()

# pickle file to store benchamrks 
with open(output_file, "wb") as fill:
    pickle.dump([par_names,parameters,combinations,comb_names,times,output_times],fill)
