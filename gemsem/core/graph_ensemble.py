'''
Efficient Graph Ensemble 
'''

import matplotlib.pyplot as plt
from time import time
import networkx as nx
import  pickle
from argparse import ArgumentParser
from scipy.io import mmread, mmwrite, loadmat
import os
import sys
import numpy as np
import pandas as pd
import json
import importlib
from sklearn import model_selection as sk_ms
import itertools
from itertools import combinations as cb


from gemben.utils      import graph_util, plot_util
from gemben.evaluation import visualize_embedding as viz
from gemben.evaluation import evaluate_graph_reconstruction as gr
from gemben.evaluation import evaluate_node_classification as e_nc
from gemben.embedding.gf       import GraphFactorization
from gemben.embedding.hope     import HOPE
from gemben.embedding.lap      import LaplacianEigenmaps
from gemben.embedding.lle      import LocallyLinearEmbedding
from gemben.embedding.node2vec import node2vec
from gemben.embedding.sdne     import SDNE


import magicgraph
from magicgraph import WeightedDiGraph, WeightedNode
sys.path.insert(0, '/home/diana/Benchmark/HARP/src/')
import graph_coarsening

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



def concat(y_s, maps):
    if not maps:
        return y_s[0]
    y_1, y_2 = y_s[-2], y_s[-1]
    map1 = maps[-1]
    y_cat = np.zeros((y_1.shape[0], y_1.shape[1] + y_2.shape[1]))
    y_cat[:, :y_1.shape[1]]  =  y_1
    for i in range(y_1.shape[0]):
        y_cat[i, y_1.shape[1]:] += y_2[map1[i], :]
    y_new = y_s[:-2]
    y_new.append(y_cat)
    return concat(y_new, maps[:-1])



def Coarsening(filename):
    G = magicgraph.load_edgelist(filename, undirected=True)
    
    G = graph_coarsening.DoubleWeightedDiGraph(G)

   
    print ('Orginal Graph')
    print ('number of nodes', G.number_of_nodes())
    print ('number of edges', G.number_of_edges())
    sfdp_path = '/home/diana/Benchmark/HARP/bin/sfdp_linux'
    Gs, mps = graph_coarsening.external_ec_coarsening(G, sfdp_path)
    return Gs, mps



def convert_nx_graph(G):
    edge_list = []
    for k,v in G.items():
        for i in v:
            edge_list.append((int(k), int(i)))
    g = nx.DiGraph()
    g.add_edges_from(edge_list)
    return g




def reformat(Gs, maps):
    map1 = maps[0]
    map1_vals = set(map1.values())
    contiguous_map = dict(zip(map1_vals, range(len(map1_vals))))
    #print(contiguous_map)
    G1 = nx.relabel_nodes(Gs[1], contiguous_map, copy=True)
    new_map = {k: contiguous_map[v] for k, v in map1.items()}
    #print(new_map)
    if len(maps) == 1:     
        return [Gs[0], G1], [new_map]
    map2 = {}
    for k, v in maps[1].items():
        map2[contiguous_map[k]] = v
        
    if len(maps) == 2:
        Gs_new_sub, maps_new_sub = reformat([G1] + Gs[2: ], [map2])
    else:
        Gs_new_sub, maps_new_sub = reformat([G1] + Gs[2: ], [map2]+ maps[2:])
    return [Gs[0]] + Gs_new_sub, [new_map] + maps_new_sub



def get_lcc(di_graph):
    di_graph = max(nx.weakly_connected_component_subgraphs(di_graph), key=len)
    tdl_nodes = di_graph.nodes()
    nodeListMap = dict(zip(tdl_nodes, range(len(tdl_nodes))))
    di_graph = nx.relabel_nodes(di_graph, nodeListMap, copy=True)
    return di_graph, nodeListMap




def get_max(val, val_max, idx, idx_max):
    if val > val_max:
        return val, idx
    else:
        return val_max, idx_max

def get_comb_embedding(methods, dims, graph_name):
    Y = None
    for i in range(len(methods)):
        m = methods[i]
        dim = dims[i]
        m_summ = '%s_%d' % (m, dim)
        res_pre = "gem/results/ensemble/%s" % graph_name
        X = pickle.load(open('%s_%s.emb' % (res_pre, m_summ), 'rb'))
        if Y is not None:
            Y = np.concatenate((Y,X), axis=1)
        else:
            Y = X
    return Y





if __name__ == '__main__':
    ''' 
    Efficient graph ensemble on Node Classification Task
    --rounds_in: rounds on evaluation on validation. only run embedding once to find best hyp
    --rounds_out: rounds on evalution on test. use the embedding from search for best hyp. 
    '''

    parser = ArgumentParser(description='Efficient Graph Ensemble Experiments')
    parser.add_argument('-data', '--graph', help='graph name')
    parser.add_argument('-exp', '--experiment', help='expriment type')
    parser.add_argument('-dims', '--dimensions', help='a list of dimensions')
    parser.add_argument('-meths', '--methods', help='a list of dimensions')
    parser.add_argument('-test_ra', '--test_ratio', help='test ratio')
    parser.add_argument('-vali_ra', '--validation_ratio', help='validation ratio in the training data')
    parser.add_argument('-rounds_in', '--rounds_in', help='rounds in validation')
    parser.add_argument('-rounds_out', '--rounds_out', help='rounds in testing')
    

    params = json.load(open('Graph_Ensemble/ens_params.conf', 'r'))
    args = vars(parser.parse_args())
    for k, v in args.items():
        if v is not None:
            params[k] = v
    params["dimensions"].sort()


    ## load dataset
    G = nx.read_gpickle('gem/data/'+params['graph']+'/graph.gpickle')
    node_labels = pickle.load(open('gem/data/'+params['graph']+'/node_labels.pickle', 'rb'), encoding = "latin1")
    G = G.to_undirected().to_directed()
    G, _ = get_lcc(G)
    nx.write_edgelist(G, params["graph"]+".edgelist")
    node_labels = node_labels[sorted(list(_.keys()))]
    print('Dataset: '+ params['graph'])
    print(nx.info(G))

    try:
        os.makedirs("gem/experiments/config/ensemble/")
    except:
        pass

    try:
        os.makedirs("gem/results/ensemble/")
    except:
        pass

    try:
        os.makedirs("gem/intermediate/ensemble/")
    except:
        pass

    try:
        os.makedirs("gem/results/ensemble_test/")
    except:
        pass


    ######## BASELINE ######
    if params['experiment'] == "baseline":
        ## split dataset into train, validation, test
        n = G.number_of_nodes()
        test_nodes = np.random.choice(n, int(float(params["test_ratio"][0])*n))
        train_nodes = list(set(G.nodes()).difference(test_nodes))


        ### 1.for each dimension, find best hyp for each method using grid search, save best conf and embedding
        ## for each individual method, save the best dim 
        ## results saved in results/emsemble
        method_best_dim = {m:0 for m in params['methods']}
        nc_method_best_dim = {m:0 for m in params['methods']}
        for dim in params['dimensions']:
            for method in params['methods']:

                best_X = None
                try:
                    model_hyp_range = json.load(
                    open('gem/experiments/config/%s_hypRange.conf' % params['graph'], 'r')
                    )
                except IOError:
                    model_hyp_range = json.load(
                    open('gem/experiments/config/default_hypRange.conf', 'r')
                    )
                ## method class 
                MethClass = getattr(
                    importlib.import_module("gem.embedding.%s" % method),
                    methClassMap[method])
                meth_hyp_range = model_hyp_range[method]
                nc_max = 0
                nc_hyp = {method: {}}
                n_r = params["rounds_in"]

                # Test each hyperparameter
                ev_cols = ["NC F1 score"]
                hyp_df = pd.DataFrame(
                columns=list(meth_hyp_range.keys()) + ev_cols + ["Round Id"]
                )
                hyp_r_idx = 0
                for hyp in itertools.product(*meth_hyp_range.values()):
                    hyp_d = {"d": dim}
                    hyp_d.update(dict(zip(meth_hyp_range.keys(), hyp)))
                    print(hyp_d)

                    if method == "sdne":
                        hyp_d.update({
                            "modelfile": [
                                "gem/intermediate/enc_mdl_%s_%d.json" % (params['graph'], dim),
                                "gem/intermediate/dec_mdl_%s_%d.json" % (params['graph'], dim)
                            ],
                            "weightfile": [
                                "gem/intermediate/enc_wts_%s_%d.hdf5" % (params['graph'], dim),
                                "gem/intermediate/dec_wts_%s_%d.hdf5" % (params['graph'], dim)
                            ]
                        })
                    elif method == "gf" or method == "node2vec":
                        hyp_d.update({"data_set": params['graph']})
                    MethObj = MethClass(hyp_d)
    
                    ##run nc experiment, test on train data, save best hyparamter
                    m_summ = '%s_%d' % (method, dim)
                    res_pre = "gem/results/ensemble/%s" % params['graph']

                    print('Learning Embedding: %s' % m_summ)

                    # X, learn_t = MethObj.learn_embedding(graph=G,
                    #                                 is_weighted=True,
                    #                                 edge_f=None,
                    #                                 no_python=True)

                    X, learn_t = MethObj.learn_embedding(graph=G,
                                                    is_weighted=False,
                                                    edge_f=None,
                                                    no_python=True)
                    print('\tTime to learn embedding: %f sec' % learn_t)
                    
                    ##test on train data, save best hyparamter
                    nc = [0] * n_r
                    nc = e_nc.expNC(X[train_nodes], 
                               node_labels[train_nodes],
                               params["validation_ratio"],
                               n_r, res_pre, m_summ)
                    print("nc", nc)
                    nc_m = np.mean(nc)
                    if nc_m >= nc_max:
                        best_X = X
                    nc_max, nc_hyp[method] = get_max(nc_m, nc_max, hyp_d, nc_hyp[method])
                    hyp_df_row = dict(zip(meth_hyp_range.keys(), hyp))
                    ## record each nc result
                    for r_id in range(n_r):
                        hyp_df.loc[hyp_r_idx, meth_hyp_range.keys()] = pd.Series(hyp_df_row)
                        hyp_df.loc[hyp_r_idx, ev_cols + ["Round Id"]] = [nc[r_id], r_id]
                        hyp_r_idx += 1
                hyp_df.to_hdf(
                    "gem/intermediate/ensemble/%s_%s_%s_%s_hyp.h5" % (params['graph'], method,params['experiment'],dim), "df")
                opt_hyp_f_pre = 'gem/experiments/config/ensemble/%s_%s_%s' % (params['graph'],method,dim)

                if nc_max and best_X is not None:
                    with open('%s_nc.conf' % opt_hyp_f_pre, 'w') as f:
                        f.write(json.dumps(nc_hyp, indent=4))
                    pickle.dump(best_X, open('%s_%s.emb' % (res_pre, m_summ), 'wb'))

                ##get best dim for each method
                if nc_max>=nc_method_best_dim[method]:
                    method_best_dim[method] = dim
                    nc_method_best_dim[method] = nc_max


                  
        print('method_best_dim : ', method_best_dim)
        print('nc_method_best_dim :', nc_method_best_dim)
        with open("gem/intermediate/ensemble/%s_%s_methodbest_dim.pickle" % (params['graph'],params['experiment']), 'wb') as fp:
            pickle.dump(method_best_dim, fp, protocol=pickle.HIGHEST_PROTOCOL)
        with open("gem/intermediate/ensemble/%s_%s_methodbest_nc.pickle" % (params['graph'],params['experiment']), 'wb') as fp:
            pickle.dump(nc_method_best_dim, fp, protocol=pickle.HIGHEST_PROTOCOL)


        with open("gem/intermediate/ensemble/%s_%s_methodbest_dim.pickle" % (params['graph'],params['experiment']), 'rb') as fp:
            method_best_dim = pickle.load(fp)
        with open("gem/intermediate/ensemble/%s_%s_methodbest_nc.pickle" % (params['graph'],params['experiment']), 'rb') as fp:
            nc_method_best_dim = pickle.load(fp)

        ###2.evalution on validation dataset with different combination of embeddings: 26*3=78 possibilities,choose best
        ### save in result/ensemble
        n_r = params["rounds_in"]
        nc_max = 0
        comb_max = []

        best_comb_df = pd.DataFrame(
                 columns= ["combination","dim","NC F1 score"])
        idx = 0
        ##add best combination: combine all methods with their best dimensions repesctively
        # for m in method_best_dim:
        #     m_summ = '%s_%d' % (m, method_best_dim[m])
        #     res_pre = "gem/results/ensemble/%s" % params['graph']
        #     X = pickle.load(open('%s_%s.emb' % (res_pre, m_summ), 'rb'))
        #     if Y is not None:
        #         Y = np.concatenate((Y,X), axis=1)
        #     else:
        #         Y = X
        ms = list(method_best_dim.keys())
        Y = get_comb_embedding(ms, [method_best_dim[m] for m in ms], params['graph'])
        Y_all_best_comb = Y
        nc = [0] * n_r
        nc = e_nc.expNC(Y[train_nodes], 
                         node_labels[train_nodes],
                         params["validation_ratio"],
                         n_r, "gem/results/ensemble/%s" % params['graph'], "alL_best_comb" )
        nc_m = np.mean(nc)
        best_comb_df.loc[idx] = [("all best combine"),0,nc_m]
        idx+=1
        nc_max, comb_max = get_max(nc_m, nc_max, ["all best combine"], comb_max)


        # for dim in params['dimensions']:
        #     ##Y: combination of 26 kinds of embedding
        #     for c in range(2,len(params['methods'])+1):
        #         for comb in cb(params['methods'], c):
        #             Y = get_comb_embedding(comb, [dim]*len(comb), params['graph'])
        #             nc = [0] * n_r
        #             nc = e_nc.expNC(Y[train_nodes], 
        #                  node_labels[train_nodes],
        #                  params["validation_ratio"],
        #                  n_r, "gem/results/ensemble/%s" % params['graph'], '_'.join(comb)+str(dim))
        #             nc_m = np.mean(nc)
        #             best_comb_df.loc[idx] = [comb,dim,nc_m]
        #             idx+=1
        #             nc_max, comb_max = get_max(nc_m, nc_max, [comb, dim], comb_max)


        ##Y: combination of all kinds of embedding
        for c in range(2,len(params['methods'])+1):
            for comb in cb(params['methods'], c):
                for dim_com in itertools.product(*[params['dimensions']]*len(comb)):
                    Y = get_comb_embedding(comb, dim_com, params['graph'])
                    nc = [0] * n_r
                    nc = e_nc.expNC(Y[train_nodes], 
                         node_labels[train_nodes],
                         params["validation_ratio"],
                         n_r, "gem/results/ensemble/%s" % params['graph'], '_'.join(comb) + '_'.join([str(i) for i in dim_com]))
                    nc_m = np.mean(nc)
                    best_comb_df.loc[idx] = [comb,dim_com,nc_m]
                    idx+=1
                    nc_max, comb_max = get_max(nc_m, nc_max, [comb, dim_com], comb_max)

        best_comb_df.to_hdf(
                 "gem/intermediate/ensemble/%s_%s_allcomb_hyp.h5" % (params['graph'],params['experiment']), "df")              
    
        ###4.evalution on test dataset with the best combination and individual method embedding with its best embedding
        ### save in results/ensemble_test
        ### individual methods 
        baseline_df = pd.DataFrame(
                 columns= ["methods","dim","NC F1 score"])
        idx = 0
        n_r = params['rounds_out']
        for m in method_best_dim.keys():
            Y = get_comb_embedding([m], [method_best_dim[m]], params['graph'])
            nc = e_nc.expNC(Y, 
                         node_labels,
                         params['test_ratio'],
                         n_r,  "gem/results/ensemble_test/%s" % params['graph'], '%s_%d' % (m, method_best_dim[m]))
            nc_m = np.mean(nc)
            baseline_df.loc[idx] = [m, method_best_dim[m],nc_m]
            idx+=1
        ### best comb
        if comb_max == ["all best combine"]:
            Y = Y_all_best_comb
        else:
            Y = get_comb_embedding(comb_max[0], comb_max[1], params['graph'])
        nc = e_nc.expNC(Y, 
                     node_labels,
                     params['test_ratio'],
                     n_r,  "gem/results/ensemble_test/%s" % params['graph'], '%s' % (str(comb_max)))
        nc_m = np.mean(nc)
        baseline_df.loc[idx] = [str(comb_max), 0,nc_m]
        baseline_df.to_hdf(
                 "gem/results/ensemble_test/%s_%s_test.h5" % (params['graph'],params['experiment']), "df")

        print("Baseline Finish!")
