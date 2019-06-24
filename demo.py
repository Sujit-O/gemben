import pickle
import networkx as nx
import pdb

map1 = pickle.load(open('gem/nodeListMap/arqui_0.pickle', 'rb'))
map2 = pickle.load(open('gem/nodeListMap/lp_lcc.pickle', 'rb'))
maps = {k: map2[v] for k,v in map1.items() if v in map2}
preds = pickle.load(open('preds.pickle', 'rb'))
G = pickle.load(open('test_graph.pickle', 'rb')

node_edges = []
for i in range(len(maps)): 
    node_edges.append([]) 
for (st, ed, w) in preds: 
    node_edges[st].append((st, ed, w))

pdb.set_trace() # Identify and define nodeName here
preds_sorted = sorted(node_edges[maps[node_name]], key=lambda x: x[2])
maps_rev = {v:k for k,v in maps.items()}
for i in range(1, 6):
    print('%s, %s' % (nodeName, maps_rev[preds_sorted[-i][1]]))
    if G.has_edge(maps[node_name], preds_sorted[-i][1]):
        print('Correct Prediction')
    else:
        print('False Prediction')
