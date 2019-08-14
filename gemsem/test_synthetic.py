import networkx as nx
import os
import subprocess
if os.name == 'posix' and 'DISPLAY' not in os.environ:
	print("Using raster graphics – high quality images using the Anti-Grain Geometry (AGG) engine")
	import matplotlib
	matplotlib.use('Agg')
	matplotlib.rcParams['text.latex.unicode']=False
import matplotlib.pyplot as plt

import seaborn
from matplotlib import rc
import numpy as np
import pdb
import random
import json

# get the modules for generating the synthetic graphs
from gemben.utils import graph_util, graph_gens
from gemben.evaluation import evaluate_graph_reconstruction as gr
from gemben.evaluation import evaluate_node_classification as e_nc
from gemben.embedding.gf       import GraphFactorization
from gemben.embedding.hope     import HOPE
from gemben.embedding.lap      import LaplacianEigenmaps
from gemben.embedding.lle      import LocallyLinearEmbedding
from gemben.embedding.node2vec import node2vec
from gemben.embedding.sdne     import SDNE

methClassMap = {"gf": "GraphFactorization",
                "hope": "HOPE",
                "lap": "LaplacianEigenmaps",
                "node2vec": "node2vec",
                "sdne": "SDNE",
                "pa": "PreferentialAttachment",
                "rand": "RandomEmb",
                "cn": "CommonNeighbors",
                "aa": "AdamicAdar",
                "jc": "JaccardCoefficient"}

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
rc('axes', titlesize='x-small')
rc('axes', linewidth=3)

plt.rc('font', **font)
seaborn.set_style("darkgrid")

# get the colormap for nodes
# TODO: find more appealing colormaps
cmap = plt.cm.get_cmap('Set1')
title = 'synthetic'

# pyvis is for interactive graph plotting
# import pyvis for visualization purpose
# from pyvis.network import Network

#add the path for the latex for macos
if os.name == 'posix':
	os.environ['PATH'] = os.environ['PATH'] + ':/Library/TeX/Root/bin/x86_64-darwin'


#############Plotting functions ###################
def get_node_color(labels):
    """Function to get the node colors for the communities. """
    node_colors = [cmap(c) for c in labels]
    return node_colors

def plot_embedding2D(node_pos, node_colors=None, di_graph=None, labels=None, shape = None):
    if shape is None:
    	embedding_dimension = node_pos[0].shape[0]
    else:
    	embedding_dimension = shape
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
        ax.title.set_text(gname[i])
        if node_labels[i]:
           node_colors = get_node_color(node_labels[i])
        else:
           node_colors = None

        plot_embedding2D(X[i], node_colors=node_colors,
	                     di_graph=di_graph[i], labels =lbl_dict[i] )
    plt.savefig('ensemble_%s.pdf' % (title), dpi=300,
                format='pdf', bbox_inches='tight')
    plt.figure()

##########Generate Synthetic graphs############
G_list =	[]
gname_list = []

# generate barabasi_albert_graph
gname_tmp = 'Barabasi Albert Graph'
G_tmp, d, dim = graph_gens.barabasi_albert_graph(100,1,0,3,'social')
G_list.append(G_tmp)
gname_list.append(gname_tmp)

# generate barabasi_albert_graph
gname_tmp = 'Random Geometric Graph'
G_tmp, _, _ = graph_gens.random_geometric_graph(100,5,0,3,'social')
G_list.append(G_tmp)
gname_list.append(gname_tmp)

# generate barabasi_albert_graph
gname_tmp = 'Stochastic Block Model Graph'
G_tmp, d, dim = graph_gens.stochastic_block_model(100,5,0,3,'social')
G_list.append(G_tmp)
gname_list.append(gname_tmp)

# generate barabasi_albert_graph
gname_tmp = 'Watts Strogatz Graph'
G_tmp = nx.watts_strogatz_graph(n=100, k=3, p=0.2)
G_list.append(G_tmp)
gname_list.append(gname_tmp)

############graph processing function##############
def process_synthetic_graphs(G_list, gname_list):
	#initialize plot parameters
	G  =[]
	pos =[]
	node_labels =[]
	lbl_dict =[]
	gname =[]

	for G_tmp,gname_tmp in zip(G_list,gname_list):
		# print(gname_tmp, G_tmp.nodes())
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

	return G, pos, node_labels, lbl_dict, gname

######################Plot the Graphs##############
G, pos, node_labels, lbl_dict, gname = process_synthetic_graphs(G_list, gname_list)

expVis(pos, gname =gname,
	node_labels=node_labels,
	di_graph=G, 
	lbl_dict=lbl_dict,
	title = 'synthetic')

#bash command to open the graph, 
#only for macos, uncomment plt.show in expVis function for linux and win 

if os.name == 'posix':
	bashCommand = "open ensemble_%s.pdf" % (title)
	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()

####################################################
# combine the graph with random edges and plot them again
def mergegraph(graphs,pos_old, labels_old, edge_prob=0.3, edge_num=0.4):
	nodes = []
	edges = []
	pos = {}
	node_cnt = 0
	val =0.9
	shift_value =[[-val,val],[val,val],[-val,-val],[val,-val]]
	for i,g in enumerate(graphs):
		tmp_nodes = list(g.nodes())
		tmp_edges = list(g.edges())

		node_map = { k:node_cnt+i for k,i in enumerate(tmp_nodes)}
		node_cnt+=len(tmp_nodes)

		new_nodes = [node_map[n] for n in tmp_nodes]
		new_edges = [(node_map[u],node_map[v]) for u,v in tmp_edges]

		#shift embedding for visual purpose
		
		for k,v in pos_old[i].items():
			pos_old[i][k][0]+=shift_value[i][0]
			pos_old[i][k][1]+=shift_value[i][1]

		new_pos = {node_map[n]:v for n,v in pos_old[i].items()}


		nodes+=new_nodes
		edges+=new_edges
		pos.update(new_pos)


	G = nx.DiGraph()
	G.add_edges_from(edges)

	# add random edges
	random.shuffle(nodes)
	l = int(edge_num*len(nodes))
	u = nodes[0:l]
	random.shuffle(nodes)
	v = nodes[0:l]

	for s, t in zip(u,v):
		if random.random()<edge_prob:
			G.add_edge(s,t)
			G.add_edge(t,s)
	nodes_deg =[G.degree[i] for i in G.nodes()]
	unq_lbl = np.unique(nodes_deg)
	lbl_map = {unq_lbl[i]:i for i in range(len(unq_lbl))}
	labels = [lbl_map[k] for k in nodes_deg]
	return G, pos, labels

# get the new graph
G, pos, labels=mergegraph(G, pos,node_labels)
# plot the new graph
colors = get_node_color(labels)

plot_embedding2D(pos, node_colors=colors, di_graph=G, labels=None, shape =2)
plt.savefig('ensemble_merged.pdf', dpi=300,
                format='pdf', bbox_inches='tight')
plt.figure()

if os.name == 'posix':
	bashCommand = "open ensemble_merged.pdf" 
	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()

# plotin spring layout
pos = nx.spring_layout(G)
plot_embedding2D(pos, node_colors=colors, di_graph=G, labels=None, shape =2)
plt.savefig('ensemble_merged_spring.pdf', dpi=300,
                format='pdf', bbox_inches='tight')

if os.name == 'posix':
	bashCommand = "open ensemble_merged_spring.pdf" 
	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()

##############start experiments with the baselines################
params =  json.load(open('./conf/ens_params.conf', 'r'))

nx.write_edgelist(G, params["graph"]+".edgelist")
node_labels = labels
print('Dataset: '+ params['graph'])
print(nx.info(G))

try:
    os.makedirs("./experiments/config/ensemble/")
except:
    pass

try:
    os.makedirs("./results/ensemble/")
except:
    pass

try:
    os.makedirs("./intermediate/ensemble/")
except:
    pass

try:
    os.makedirs("./results/ensemble_test/")
except:
    pass

#TODO1: Get the embedding and node classification result from each baseline Separately
#TODO2: Get the embedding and node classificationr result after ensemble

