'''
====================================================
Module to run a single or multiple examples
====================================================
Module to run the benchmark across all the baseline embedding algorithms.
'''

from subprocess import call
import itertools
try: import cPickle as pickle
except: import pickle
import json
import networkx as nx
import pandas as pd
import pdb
import os
import sys
from time import time
from gemben.utils import graph_gens

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

class exp:
    def __init__(self, domain="social", method="sdne", rounds=1,lexp=False, samp_scheme='rw', plot_hyp_data=False):
        
        t1 = time()
        self.params = json.load(
            open('gemben/experiments/config/params_benchmark.conf', 'r')
        )
        self.domain_graph_map = json.load(
            open('gemben/experiments/config/domain_graph_map.conf', 'r')
        )
        # graph_hyp_range: {N: [128, 256, 512, 1024], deg: [4, 6,8, 10, 12]}
        self.graph_hyp_range = json.load(
            open('gemben/experiments/config/graph_hyp_range.conf', 'r')
        )
        # def_graph_hyps: {N: 1024, deg: 8, dia: None, dim: 128}
        self.def_graph_hyps = json.load(
            open('gemben/experiments/config/def_graph_hyps.conf', 'r')
        )

        self.params["rounds"] = rounds
        self.params["graphs"] = self.domain_graph_map[domain]
        self.params["lexp"] = lexp
        self.params["plot_hyp_data"] = plot_hyp_data
        if method == "all":
            self.params["methods"] = methClassMap.keys()
        elif len(method)>1:
            self.params["methods"] = method.split(',')
        else:
            self.params["methods"] = self.method
        self.samp_scheme = samp_scheme


    def run(self):    

        try:
          os.makedirs("gemben/intermediate")
        except:
          pass
        try:
          os.makedirs("gemben/results")
        except:
          pass
        try:
          os.makedirs("gemben/temp")
        except:
          pass

        graph_hyp_keys = list(self.graph_hyp_range.keys())
        ev_cols = ["LP MAP", "LP P@100"]
        for meth , graph in itertools.product(*[self.params["methods"],self.params["graphs"]]):
            hyp_df = pd.DataFrame(
                    columns=graph_hyp_keys + ev_cols + ["Round Id"]
                )
            hyp_r_idx = 0
            for hyp_key in graph_hyp_keys:
               
                for curr_hyp_key_range, r_id in itertools.product(
                    *[graph_hyp_range[hyp_key], range(self.params["rounds"])]
                ):
                    
                    if r_id == 0: 
                        f_hyp = 1
                    else:
                        f_hyp = 0
                    
                    curr_hyps = self.def_graph_hyps.copy()
                
                    curr_hyps[hyp_key] = curr_hyp_key_range
                    curr_hyps["domain"] = self.params["domain_name"]
                    hyp_str = '_'.join(
                                "%s=%s" % (key, str(val).strip("'")) for (key, val) in curr_hyps.items()
                            )
                    
                    hyp_str_graph_name = '_'.join(
                                "%s=%s" % (key, str(val).strip("'")) for (key, val) in curr_hyps.items() if key != 'dim' 
                            )
                    
                    syn_data_folder = 'benchmark_%s_%s_%s' % (graph, hyp_str_graph_name, r_id)
                    
                    graphClass = getattr(graph_gens, graph)
                    
                    try:
                        nx.read_gpickle(
                                'gemben/data/%s/graph.gpickle' % syn_data_folder
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
                            if not os.path.exists("gemben/data/%s" % syn_data_folder):
                                os.makedirs("gemben/data/%s" % syn_data_folder)
                            nx.write_gpickle(
                                    G, 'gemben/data/%s/graph.gpickle' % syn_data_folder
                        )
                    perf_exp = not self.params["lexp"]      
                    if self.params["lexp"]:
                      try:
                        MAP, prec, n_samps = pickle.load(
                          open('gemben/results/%s_%s_%d_%s.lp' % (
                              syn_data_folder, meth, 
                              curr_hyps["dim"], self.samp_scheme), 'rb'))
                      except:   
                          perf_exp = 1
                           ##### only find the best hyp for first round
                    if perf_exp:
                        os.system(
                          "python3 gemben/experiments/exp.py -data %s -meth %s -dim %d -rounds 1 -find_hyp %d -s_sch %s -exp lp" % (
                              syn_data_folder,
                              meth,
                              curr_hyps["dim"],
                              f_hyp,
                              self.samp_scheme
                          )
                        )
                    MAP, prec, n_samps = pickle.load(
                    open('gemben/results/%s_%s_%d_%s.lp' % (
                        syn_data_folder, meth, 
                        curr_hyps["dim"], self.samp_scheme), 'rb'))        
                    hyp_df.loc[hyp_r_idx, graph_hyp_keys] = \
                    pd.Series(curr_hyps)
                    #prec_100 = prec[int(n_samps[0])][0][100]
                    try:
                      prec_100 = list(prec.values())[0][0][100]
                    except:
                      pdb.set_trace()
                    f_temp = open("gemben/temp/%s_%s_%s_lp_%s_data_hyp.txt" % (
                      self.params["domain_name"], graph, meth, self.samp_scheme), 'a')
                    f_temp.write('%s: round: %d, MAP: %f, prec_100: %f' % (hyp_str, r_id, list(MAP.values())[0][0], prec_100))
                    f_temp.close()
                    hyp_df.loc[hyp_r_idx, ev_cols + ["Round Id"]] = \
                    [list(MAP.values())[0][0], prec_100, r_id]
                    #[MAP[int(n_samps[0])][0], prec_100, r_id]
                    hyp_r_idx += 1

            hyp_df.to_hdf(
                "gemben/intermediate/%s_%s_%s_lp_%s_data_hyp.h5" % (
                    self.params["domain_name"], graph, meth, self.samp_scheme),
                "df"
            )
            print('Experiments done for %s, %s' % (graph, meth))
