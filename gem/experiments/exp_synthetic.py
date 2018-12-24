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
sys.path.insert(0, './')
from gem.utils import plot_util

if __name__ == "__main__":
    samp_scheme = "u_rand"
    meths = ["gf", "rand", "pa", "hope", "lap"]
    load_res = False
    n_rounds = 5
    syn_hyps = json.load(
        open('gem/experiments/config/syn_hypRange.conf', 'r')
    )
    for syn_data in syn_hyps.keys():
        syn_hyp_range = syn_hyps[syn_data]
        hyp_keys = syn_hyp_range.keys()
        graphClass = getattr(nx, syn_data)
        ev_cols = ["GR MAP", "LP MAP", "NC F1 score"]
        for meth in meths:
            hyp_df = pd.DataFrame(
                columns=hyp_keys + ev_cols + ["Round Id"]
            )
            hyp_r_idx = 0
            for hyp in itertools.product(*syn_hyp_range.values()):
                hyp_dict = dict(zip(hyp_keys, hyp))
                hyp_str = '_'.join("%s=%r" % (key,val) for (key,val) in hyp_dict.iteritems())
                syn_data_folder = 'synthetic/%s_%s' % (syn_data, hyp_str)
                hyp_df_row = dict(zip(hyp_keys, hyp))
                for r_id in range(n_rounds):
                    G = graphClass(**hyp_dict)
                    if not os.path.exists("gem/data/%s" % syn_data_folder):
                        os.makedirs("gem/data/%s" % syn_data_folder)
                    nx.write_gpickle(G, 'gem/data/%s/graph.gpickle' % syn_data_folder)
                    os.system(
                        "python gem/experiments/exp.py -data %s -find_hyp 1 -meth %s -dim 2 -s_sch %s -exp lp" % (syn_data_folder, meth, samp_scheme)
                    )
                    MAP, prec, n_samps = pickle.load(
                        open('gem/results/%s_%s_2_%s.lp' % (syn_data_folder, meth, samp_scheme), 'rb')
                    )        
                    hyp_df.loc[hyp_r_idx, hyp_keys] = \
                        pd.Series(hyp_df_row)
                    hyp_df.loc[hyp_r_idx, ev_cols + ["Round Id"]] = \
                        [0, MAP[int(n_samps[0])][0], 0, r_id]
                    hyp_r_idx += 1
            hyp_df.to_hdf(
                "gem/intermediate/%s_%s_lp_%s_data_hyp.h5" % (syn_data, meth, samp_scheme),
                "df"
            )
        plot_util.plot_hyp_data(hyp_keys, ["lp"], meths, syn_data, samp_scheme)

