from subprocess import call
import itertools
try: import cPickle as pickle
except: import pickle
import json
from argparse import ArgumentParser
import networkx as nx
import pandas as pd
import pdb
import os
import sys
from time import time
sys.path.insert(0, './')
from gem.utils import graph_gens

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


if __name__ == "__main__":
    ''' Sample usage
    python experiments/exp_synthetic.py -syn_names all -plot_hyp_data 1 -meths all
    '''
    t1 = time()
    parser = ArgumentParser(description='Graph Embedding Benchmark Experiments')
    parser.add_argument('-domain', '--domain_name',
                        help='domain name (default: social)')
    parser.add_argument('-graph', '--graphs',
                        help='graph name (default: all)')
    parser.add_argument('-meth', '--methods',
                        help='method list (default: all)')
    parser.add_argument('-plot_hyp_data', '--plot_hyp_data',
                        help='plot the hyperparameter results (default: False)')
    parser.add_argument('-rounds', '--rounds',
                        help='number of rounds (default: 20)')
    parser.add_argument('-s_sch', '--samp_scheme',
                        help='sampling scheme (default: rw)')
    parser.add_argument('-lexp', '--lexp',
                        help='load experiment (default: False)')
    params = json.load(
        open('gem/experiments/config/params_benchmark.conf', 'r')
    )
    args = vars(parser.parse_args())
    print (args)
    domain_graph_map = json.load(
        open('gem/experiments/config/domain_graph_map.conf', 'r')
    )
    # graph_hyp_range: {N: [128, 256, 512, 1024], deg: [4, 6,8, 10, 12]}
    graph_hyp_range = json.load(
        open('gem/experiments/config/graph_hyp_range.conf', 'r')
    )
    # def_graph_hyps: {N: 1024, deg: 8, dia: None, dim: 128}
    def_graph_hyps = json.load(
        open('gem/experiments/config/def_graph_hyps.conf', 'r')
    )
    for k, v in args.items():
        if v is not None:
            params[k] = v
    params["rounds"] = int(params["rounds"])
    #params["domain_name"] = params["domain_name"].split(',')
    if params["graphs"] == "all":
        params["graphs"] = domain_graph_map[params["domain_name"]]
    else:
        params["graphs"] = params["graphs"].split(',')
    params["lexp"] = bool(int(params["lexp"]))
    params["plot_hyp_data"] = bool(int(params["plot_hyp_data"]))
    if params["methods"] == "all":
        params["methods"] = methClassMap.keys()
    else:
        params["methods"] = params["methods"].split(',')
    samp_scheme = params["samp_scheme"]





    

    try:
      os.makedirs("gem/intermediate")
    except:
      pass
    try:
      os.makedirs("gem/results")
    except:
      pass
#     if not os.path.exists("gem/intermediate"):
#         os.makedirs("gem/intermediate")
#     if not os.path.exists("gem/results"):
#         os.makedirs("gem/results")

    graph_hyp_keys = list(graph_hyp_range.keys())
    ev_cols = ["LP MAP", "LP P@100"]
    for meth , graph in itertools.product(*[params["methods"],params["graphs"]]):
        hyp_df = pd.DataFrame(
                columns=graph_hyp_keys + ev_cols + ["Round Id"]
            )
        hyp_r_idx = 0
        for hyp_key in graph_hyp_keys:
           
            for curr_hyp_key_range, r_id in itertools.product(
                *[graph_hyp_range[hyp_key], range(params["rounds"])]
            ):
                
                
                ##### first round to find the best parameter for each methods
                if r_id == 0: 
                    f_hyp = 1
                else:
                    f_hyp = 0
                
                
                
                curr_hyps = def_graph_hyps.copy()
            
                curr_hyps[hyp_key] = curr_hyp_key_range
                curr_hyps["domain"] = params["domain_name"]
                hyp_str = '_'.join(
                            "%s=%s" % (key, str(val).strip("'")) for (key, val) in curr_hyps.items()
                        )
                syn_data_folder = 'benchmark_%s_%s_%s' % (graph, hyp_str, r_id)
                
                graphClass = getattr(graph_gens, graph)
                
                try:
                    nx.read_gpickle(
                            G, 'gem/data/%s/graph.gpickle' % syn_data_folder
                  )
                except:
                    flag =  1
                    ##### flag = 0 means the labels are continous on lcc
                    while flag:
                        print("Graph is generating...")
                        G = graphClass(**curr_hyps)[0]
                        if len(set(G.nodes())) == G.number_of_nodes() and list(G.nodes())[-1] == G.number_of_nodes() -1:
                            flag = 0
                    if G:
                        if not os.path.exists("gem/data/%s" % syn_data_folder):
                            os.makedirs("gem/data/%s" % syn_data_folder)
                        nx.write_gpickle(
                                G, 'gem/data/%s/graph.gpickle' % syn_data_folder
                    )
                    
                    ##### only find the best hyp for first round
                os.system(
                  "python3 gem/experiments/exp.py -data %s -meth %s -dim %d -rounds 1 -find_hyp %d -s_sch %s -exp lp" % (
                      syn_data_folder,
                      meth,
                      curr_hyps["dim"],
                      f_hyp,
                      samp_scheme
                  )
                )
                MAP, prec, n_samps = pickle.load(
                open('gem/results/%s_%s_%d_%s.lp' % (
                    syn_data_folder, meth, 
                    curr_hyps["dim"], samp_scheme), 'rb'))        
                hyp_df.loc[hyp_r_idx, graph_hyp_keys] = \
                pd.Series(curr_hyps)
                prec_100 = prec[int(n_samps[0])][0][100]
                hyp_df.loc[hyp_r_idx, ev_cols + ["Round Id"]] = \
                [MAP[int(n_samps[0])][0], prec_100, r_id]
                hyp_r_idx += 1

        hyp_df.to_hdf(
            "gem/intermediate/%s_%s_%s_lp_%s_data_hyp.h5" % (
                params["domain_name"], graph, meth, samp_scheme),
            "df"
        )
