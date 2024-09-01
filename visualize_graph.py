import numpy as np
import matplotlib.pyplot as plt

fontsize = 12

# Box size in comoving kpc/h 
boxsize = 1.e6

# colors
col_1 = '#648FFF'
col_2 = '#785EF0'
col_3 = '#DC267F'
col_4 = '#FE6100'
col_5 = '#FFB000'

def visualize_graph(num, pos_masses, pars_file, edge_index):

    fig = plt.figure(figsize=(12, 12))

    ax = fig.add_subplot(projection ="3d")
    pos = pos_masses[:,:3]
    masses = pos_masses[:,3]

    pos *= boxsize/1.e3   # show in Mpc

    # Draw lines for each edge
    for (src, dst) in edge_index: #.t().tolist():

        src = pos[int(src)].tolist()
        dst = pos[int(dst)].tolist()

        ax.plot([src[0], dst[0]], [src[1], dst[1]], zs=[src[2], dst[2]], linewidth=0.6, color='dimgrey')

    # Plot nodes
    mass_mean = np.mean(masses)
    for i,m in enumerate(masses):
            ax.scatter(pos[i, 0], pos[i, 1], pos[i, 2], s=50*m*m/(mass_mean**2), zorder=1000, alpha=0.6, color = 'mediumpurple')

    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)
    ax.zaxis.set_tick_params(labelsize=fontsize)

    ax.set_xlabel('x (Mpc)', fontsize=16, labelpad=15)
    ax.set_ylabel('y (Mpc)', fontsize=16, labelpad=15)
    ax.set_zlabel('z (Mpc)', fontsize=16, labelpad=15)

    rl = '$R_{link} = 0.2$'

    ax.set_title(f'\tGraph n°{num}, Masses $\\geq 99.7$% percentile, {rl} Mpc \t \n \n $\\Omega_m = {float(pars_file[0]):.3f}$ \t $\\sigma_8 = {float(pars_file[1]):.3f}$', fontsize=20)

    # fig.savefig("Plots/Graphs/graph_"+num+"_997.png", bbox_inches='tight', pad_inches=0.6, dpi=400)
    # plt.close(fig)

    plt.show()