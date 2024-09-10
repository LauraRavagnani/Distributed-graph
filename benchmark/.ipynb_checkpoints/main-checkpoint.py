import readfof
from pyspark.sql import SparkSession
import numpy as np
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







output_file = 'time_output.pkl'
memories_1 = ['1g', '2g', '4g', '8g']
cuts_2 = [0.995, 0.996, 0.997, 0.998]
partitions_3 = [4, 8, 16, 32]
cores_4 = [1,2,3,4]

combinations = 6 #binomiale con n = 4 e k = 2 (4 parametri da cambiare una coppia alla volta)

# Default combinations (x: y - z) :
# 0: 1 - 2
# 1: 1 - 3
# 2: 1 - 4
# 3: 2 - 3
# 4: 2 - 4
# 5: 3 - 4

output = numpy.empty(shape = (6,4,4), dtype=double)

#0: 4g - 0.998 equal to (0,2,3)
for memory in ['4g']:#['1g', '2g', '4g', '8g']
    for cut in [0.998]: #[0.995, 0.996, 0.997, 0.998]
        
        ###################### spark context ######################

        spark = SparkSession.builder \
        .master("spark://master:7077")\
        .appName("CosmoSparkApplicationBenchmark")\
        .config("spark.executor.memory", memory)\
        .getOrCreate()
        sc = spark.sparkContext   


        start_time = time.time()
        print('--------------------------------------')
        print('spark context enable')
        print('memory :', memory, ' cut :', cut, '\n')
        start_time = time.time()


        # number of simulations to be processed
        n_sims = 1000

        # path list with simulation keys
        path_list = [(i, "/mnt/cosmo_GNN/Data/" + str(i)) for i in range(n_sims)]

        # parallelize path list and read files
        fof_rdd = sc.parallelize(path_list)\
                    .mapValues(read_cosmo_data)
        
        # get positions and masses for each point
        pos_mass_rdd = fof_rdd.mapValues(get_pos_mass)\
                              .flatMap(assign_key_to_rows)
        # cut percentile
        cut = cut

        # get mass cuts 
        mass_cut_rdd = fof_rdd.mapValues(get_pos_mass)\
                              .mapValues(lambda x: np.quantile(x[:, -1], cut))
        
        print('collecting cut values ', )
        cut_start = time.time()
        mass_cuts = mass_cut_rdd.values().collect()
        cut_end = time.time()
        print('\r', cut_end-cut_start, '\n')

        mass_cuts = np.array(mass_cuts)

        # filter by mass
        pos_mass_rdd_filtered = pos_mass_rdd.filter(lambda x: x[1][-1] >= mass_cuts[x[0]])

        boxes = sub_box_bounds(5,0.2)

        # masses rdd ---> (simkey, mass)
        mass_rdd = pos_mass_rdd_filtered.mapValues(lambda x: x[3])

        # positions rdd ---> (simkey, pos)
        pos_rdd = pos_mass_rdd_filtered.mapValues(lambda x: x[:3])

        # indexed positions rdd (point indexes)
        # --> ( simkey, (point_idx, array(x, y, z)) )
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
        diff_rdd = cartesian_rdd.map(lambda x:(x[0][0],(x[0][1][0], x[1][1][0], x[0][1][1],  x[1][1][1] , x[0][1][1] - x[1][1][1])))

        # --> (simkey_boxkey, (idx1, idx2, coord1, coord2, diff_coord, norm))
        pairs_dist_rdd_with_box = diff_rdd.mapValues(lambda x: (x[0], x[1], x[2], x[3], x[4], np.linalg.norm(x[4])))

        pairs_dist_rdd_no_box = pairs_dist_rdd_with_box.map(lambda x: (int(x[0].split('_')[0]), (x[1])))\
                                                       .map(convert_to_tuple)\
                                                       .distinct()\
                                                       .map(convert_to_array)
        
        linked_pairs_dist_rdd = pairs_dist_rdd_no_box.filter(lambda x: x[1][-1] <= 0.2)

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

        print('counting halos ')
        halos_start = time.time()
        n_halos = pos_mass_rdd_filtered.countByKey()
        halos_end = time.time()
        print('\r',halos_end-halos_start, '\n')

        u = sc.parallelize([(i[0], math.log10(i[1])) for i in n_halos.items()])

        # graph rdd (a graph for each simulation)
        # masses, positions, simulation parameters, global features, edge indexes, edge features
        print('joining rdds ')
        joining_start = time.time()
        raw_graph_rdd = grouped_idx_pos_rdd.join(grouped_edge_rdd)\
                                           .join(u)\
                                           .join(param_rdd)\
                                           .mapValues(lambda x: (x[0][0][0], x[0][0][1], x[0][1], x[1]))
        joining_end = time.time()
        print('\r', joining_end - joining_start, '\n')

        graph_rdd = raw_graph_rdd.mapValues(lambda x: create_graph(x))   

        print('collecting graphs ')
        graph_start = time.time()
        graph_rdd.collect()
        graph_stop = time.time()
        print('\r', graph_stop-graph_start, '\n')


        end_time = time.time()

        print('\ntime for graphs: ', end_time - start_time, '\n')

        
        sc.stop()
        spark.stop()

with open(output_file, "wb") as fill:
    pickle.dump(output, fill)
