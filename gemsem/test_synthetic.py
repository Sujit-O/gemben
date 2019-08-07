import networkx as nx
import matplotlib.pyplot as plt
import os

# import pyvis for visualization purpose
from pyvis.network import Network

import sys
sys.path.append('./')
sys.path.append(os.path.realpath(__file__))

# get the modules for generating the synthetic graphs
from gem.utils import graph_util, graph_gens

##########Generate Barabasi-Albert Graph############
# generate barabasi_albert_graph
G_barabasi = graph_gens.barabasi_albert_graph(100)

#initialize the pyvis network graph
net = Network()

#covert the networkx graph to pyvis network
net.from_nx(G_barabasi)

#enable physics
net.enable_physics(True)
#draw the graph
net.show("barabasi_graph.html")
plt.show()

####################################################



