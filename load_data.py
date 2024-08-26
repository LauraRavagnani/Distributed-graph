import scipy.spatial as SS
import readfof
import numpy as np

# Normalize QUIJOTE parameters
def normalize_params(params):

    minimum = np.array([0.1, 0.5])
    maximum = np.array([0.5, 2.0])
    params = (params - minimum)/(maximum - minimum)
    return params

# KDTree: provides an index into a set of k-dimensional points 
# which can be used to rapidly look up the nearest neighbors of any point

# Compute KDTree and get edges and edge features
def get_edges(pos, r_link):

    # 1. Get edges

    # Create the KDTree and look for pairs within a distance r_link (clustering phase)

    # Boxsize normalized to 1
    kd_tree = SS.KDTree(
        pos,                # data 
        leafsize = 16,      # threshold at which the algorithm stops splitting 
                            # and directly stores points in a leaf node
                            # threshold on the number of points
        boxsize = 1.0001    # apply a m-d toroidal topology to the KDT 
                            # --> periodic boundary condition
                            # but small tolerance (small value for boxsize)
                            # --> the tree accounts for points very close 
                            # to the boundary, improving the accuracy of neighbor 
                            # and distance calculations in a periodic space
        )
    
    # Find all pairs of points in the KDT whose distance is at most r (maximum distance)
    # returns point indexes
    edge_index = kd_tree.query_pairs(r=r_link, output_type="ndarray") 

    # Ensure that for every pair of points found within r_link, 
    # the reverse pair is also included --> symmetry
    reversepairs = np.zeros((edge_index.shape[0],2))
    for i, pair in enumerate(edge_index):
        reversepairs[i] = np.array([pair[1], pair[0]])
    edge_index = np.append(edge_index, reversepairs, 0)

    # indexes must be integers
    edge_index = edge_index.astype(int)

    # Write in pytorch-geometric format
    edge_index = edge_index.T
    num_pairs = edge_index.shape[1]

    # 2. Get edge attributes
    row, col = edge_index

    # Calculating distance between linked halo pairs
    diff = pos[row]-pos[col]

    # Taking into account periodic boundary conditions
    diff_bc = np.where(diff < -0.01, diff + 1.0, diff)
    diff = np.where(diff > 0.01, diff - 1.0, diff_bc)

    # Get translational and rotational invariant features

    # Distance d = sqrt(dx^2+dy^2+dz^2)
    dist = np.linalg.norm(diff, axis=1) 

    # Centroid of halo catalogue (3d position of the centroid)
    centroid = np.mean(pos,axis=0)

    # Vectors of node and neighbor --> ??
    # distance between each point and the centroid
    row = (pos[row] - centroid)
    col = (pos[col] - centroid)

    # Taking into account periodic boundary conditions
    row_bc = np.where(row < -0.5, row + 1, row)
    row = np.where(row > 0.5, row - 1, row_bc)

    col_bc = np.where(col < -0.5, col + 1, col)
    col = np.where(col > 0.5, col - 1, col_bc)

    # Normalizing
    unitrow = row/(np.linalg.norm(row, axis = 1).reshape(-1, 1))  
    unitcol = col/(np.linalg.norm(col, axis = 1).reshape(-1, 1))
    unitdiff = diff/(dist.reshape(-1,1))

    # Dot products between unit vectors
    cos1 = np.array([np.dot(unitrow[i,:].T,unitcol[i,:]) for i in range(num_pairs)])
    cos2 = np.array([np.dot(unitrow[i,:].T,unitdiff[i,:]) for i in range(num_pairs)])

    # Normalize distance by linking radius
    dist /= r_link

    # Concatenate to get all edge attributes
    edge_attr = np.concatenate([dist.reshape(-1,1), cos1.reshape(-1,1), cos2.reshape(-1,1)], axis=1)

    # Self loops (self interactions)
    loops = np.zeros((2,pos.shape[0]),dtype=int)
    atrloops = np.zeros((pos.shape[0],3))
    for i, _ in enumerate(pos):
        loops[0,i], loops[1,i] = i, i
        atrloops[i,0], atrloops[i,1], atrloops[i,2] = 0., 1., 0.
    edge_index = np.append(edge_index, loops, 1)
    edge_attr = np.append(edge_attr, atrloops, 0)
    
    edge_index = edge_index.astype(int)

    return edge_index, edge_attr

# Routine to create a cosmic graph from a halo catalogue
def sim_graph(simnumber, filename, paramsfile):

    # Get some hyperparameters
    r_link = 0.2
    pred_params = 1

    # Read Fof
    FoF = readfof.FoF_catalog(
        filename,           # simulation file name
        2,                  # snapnum, indicating the redshift (z=1)
        long_ids = False,
        swap = False,
        SFR = False,
        read_IDs = False
        )
    
    # Get positions and masses
    pos = FoF.GroupPos/1e06             # Halo positions in Gpc/h 
    mass_raw = FoF.GroupMass * 1e10     # Halo masses in Msun/h

    # Mass cut
    cut_val = np.quantile(mass_raw,0.997)    # universal mass cut
    mass_mask = (mass_raw >= cut_val)
    mass_mask = mass_mask.reshape(-1) # CHECK
    mass = mass_raw[mass_mask]  
    pos = pos[mass_mask]    

   
    # Get the output to be predicted by the GNN
    # Read the value of the cosmological parameters
    params = np.array(paramsfile[simnumber],dtype=np.float32)
    # Normalize true parameters
    params = normalize_params(params)
    # Consider the correct number of parameters
    params = params[:pred_params]   
    y = np.reshape(params, (1,params.shape[0]))
    
    # Number of halos as global features
    u = np.log10(pos.shape[0]).reshape(1,1) 
    # Nodes features
    x = torch.tensor(mass, dtype=torch.float32)
    # Get edges and edge features
    edge_index, edge_attr = get_edges(pos, r_link)
    # Construct the graph
    graph = Data(
        x = x.resize_(x.size()[0],1),                               # node feature
        y = torch.tensor(y, dtype=torch.float32),                   # true label
        u = torch.tensor(u, dtype=torch.float32),                   # global features
        edge_index = torch.tensor(edge_index, dtype=torch.long),    # graph connectivity
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)    # edge feature matrix
        ) 
    
    return graph, cut_val