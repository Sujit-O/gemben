import networkx as nx
import os
import subprocess
if os.name == 'posix' and 'DISPLAY' not in os.environ:
	print("Using raster graphics – high quality images using the Anti-Grain Geometry (AGG) engine")
	import matplotlib
	matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn
from matplotlib import rc
import numpy as np
import pdb

# set the searborn and matplotlib formatting style
font = {'family': 'serif', 'serif': ['computer modern roman']}
rc('text', usetex=True)
rc('font', weight='bold')
rc('font', size=20)
rc('lines', markersize=10)
rc('xtick', labelsize=5)
rc('ytick', labelsize=5)
rc('axes', labelsize='x-large')
rc('axes', labelweight='bold')
rc('axes', titlesize='x-large')
rc('axes', linewidth=3)
plt.rc('font', **font)
seaborn.set_style("darkgrid")

cmap = plt.cm.get_cmap('Set1')
title = 'synthetic'
# import pyvis for visualization purpose
# from pyvis.network import Network

# get the modules for generating the synthetic graphs
from gemben.utils import graph_util, graph_gens

#############Plotting functions ###################
def get_node_color(labels):
    """Function to get the node colors for the communities. """
    node_colors = [cmap(c) for c in labels]
    return node_colors

def plot_embedding2D(node_pos, node_colors=None, di_graph=None, labels=None):
    embedding_dimension = node_pos[0].shape[0]
    if (embedding_dimension > 2):
        print("Embedding dimension greater than 2, use tSNE to reduce it to 2")
        model = TSNE(n_components=2)
        node_pos = model.fit_transform(node_pos)

    if di_graph is None:
        # plot using plt scatter
        plt.scatter(node_pos[:, 0], node_pos[:, 1], c=node_colors)
    else:
        # plot using networkx with edge structure
        if node_colors:
            nodes_draw = nx.draw_networkx_nodes(di_graph, node_pos,
                                                node_color=node_colors,
                                                width=0.2, node_size=40,
                                                arrows=False, alpha=0.9,
                                                font_size=5)
            nodes_draw.set_edgecolor('w')
        else:
            nodes_draw = nx.draw_networkx(di_graph, pos, node_color=node_colors,
                                          width=0.2, node_size=40, arrows=False,
                                          alpha=0.9, font_size=12)
            nodes_draw.set_edgecolor('w')
        nx.draw_networkx_edges(di_graph,node_pos,arrows=False,width=0.4,alpha=0.8,edge_color='#6B6B6B')
        # nx.draw_networkx_labels(di_graph, pos=node_pos, labels= labels)   

def expVis(X, gname='test', node_labels=None, di_graph=None, lbl_dict=None, title = 'test'):
    print('\tGraph Visualization:')
    pos =1
    for i in range(len(gname)):
        ax= plt.subplot(220 + pos)
        pos += 1
        # ax.title.set_text(gname[i])
        if node_labels[i]:
           node_colors = get_node_color(node_labels[i])
        else:
           node_colors = None

        plot_embedding2D(X[i], node_colors=node_colors,
	                     di_graph=di_graph[i], labels =lbl_dict[i] )
    plt.savefig('ensemble_%s.pdf' % (title), dpi=300,
                format='pdf', bbox_inches='tight')
    plt.figure()

####################Initialize Plot Params##########
G  =[]
pos =[]
node_labels =[]
lbl_dict =[]
gname =[]

##########Generate Barabasi-Albert Graph############
# generate barabasi_albert_graph
gname_tmp = 'barabasi'

G_tmp, d, dim = graph_gens.barabasi_albert_graph(100,1,0,3,'social')
print(gname_tmp, G_tmp.nodes())
pos_tmp = nx.spring_layout(G_tmp)
nodes_deg =[G_tmp.degree[i] for i in G_tmp.nodes()]
unq_lbl = np.unique(nodes_deg)
lbl_map = {unq_lbl[i]:i for i in range(len(unq_lbl))}
lbl_map_rev = {v:k for k,v in lbl_map.items()}

node_labels_tmp = [lbl_map[k] for k in nodes_deg]
lbl_dict_tmp = {n:i for n,i in enumerate(G_tmp.nodes())}

G.append(G_tmp)
pos.append(pos_tmp)
node_labels.append(node_labels_tmp)
lbl_dict.append(lbl_dict_tmp)
gname.append(gname_tmp)

##########Generate Barbell Graph############
# generate barabasi_albert_graph
gname_tmp = 'random'

G_tmp, _, _ = graph_gens.random_geometric_graph(100,5,0,3,'social')
print(gname_tmp, G_tmp.nodes())
pos_tmp = nx.spring_layout(G_tmp)
nodes_deg =[G_tmp.degree[i] for i in G_tmp.nodes()]
unq_lbl = np.unique(nodes_deg)
lbl_map = {unq_lbl[i]:i for i in range(len(unq_lbl))}
lbl_map_rev = {v:k for k,v in lbl_map.items()}

node_labels_tmp = [lbl_map[k] for k in nodes_deg]
lbl_dict_tmp = {n:i for n,i in enumerate(G_tmp.nodes())}

G.append(G_tmp)
pos.append(pos_tmp)
node_labels.append(node_labels_tmp)
lbl_dict.append(lbl_dict_tmp)
gname.append(gname_tmp)

##########Generate Barabasi-Albert Graph############
# generate barabasi_albert_graph
gname_tmp = 'sbm'

G_tmp, d, dim = graph_gens.stochastic_block_model(100,5,0,3,'social')
print(gname_tmp, G_tmp.nodes())
pos_tmp = nx.spring_layout(G_tmp)
nodes_deg =[G_tmp.degree[i] for i in G_tmp.nodes()]
unq_lbl = np.unique(nodes_deg)
lbl_map = {unq_lbl[i]:i for i in range(len(unq_lbl))}
lbl_map_rev = {v:k for k,v in lbl_map.items()}

node_labels_tmp = [lbl_map[k] for k in nodes_deg]
lbl_dict_tmp = {n:i for n,i in enumerate(G_tmp.nodes())}

G.append(G_tmp)
pos.append(pos_tmp)
node_labels.append(node_labels_tmp)
lbl_dict.append(lbl_dict_tmp)
gname.append(gname_tmp)

##########Generate Barabasi-Albert Graph############
# generate barabasi_albert_graph
gname_tmp = 'watts_strogatz_graph'

G_tmp = nx.watts_strogatz_graph(n=100, k=3, p=0.2)
print(gname_tmp, G_tmp.nodes())
pos_tmp = nx.spring_layout(G_tmp)
nodes_deg =[G_tmp.degree[i] for i in G_tmp.nodes()]
unq_lbl = np.unique(nodes_deg)
lbl_map = {unq_lbl[i]:i for i in range(len(unq_lbl))}
lbl_map_rev = {v:k for k,v in lbl_map.items()}

node_labels_tmp = [lbl_map[k] for k in nodes_deg]
lbl_dict_tmp = {n:i for n,i in enumerate(G_tmp.nodes())}

G.append(G_tmp)
pos.append(pos_tmp)
node_labels.append(node_labels_tmp)
lbl_dict.append(lbl_dict_tmp)
gname.append(gname_tmp)
######################Plot the Graphs##############

expVis(pos, gname =gname,
	node_labels=node_labels,
	di_graph=G, 
	lbl_dict=lbl_dict,
	title = 'synthetic')

#bash command to open the graph, 
#only for macos, uncomment plt.show for linux and win 

bashCommand = "open ensemble_%s.pdf" % (title)
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

####################################################

