'''
=========================
Plotting Results
=========================
Code example to plot the results after the experiment.
'''

try: import cPickle as pickle
except: import pickle

import matplotlib
import matplotlib.pyplot as plt
import itertools
from matplotlib import rc
import numpy as np
import pandas as pd
import seaborn

font = {'family': 'serif', 'serif': ['computer modern roman']}
rc('text', usetex=False)
rc('font', weight='bold')
rc('font', size=8)
rc('lines', markersize=2.5)
rc('lines', linewidth=0.5)
rc('xtick', labelsize=6)
rc('ytick', labelsize=6)
rc('axes', labelsize='small')
rc('axes', labelweight='bold')
rc('axes', titlesize='small')
rc('axes', linewidth=1)
plt.rc('font', **font)
seaborn.set_style("darkgrid")
import pdb


def plot_benchmark(domains, graph_attrs, attr_defaults, methods, graph_names, s_sch='rw'):
    df_all = pd.DataFrame()
    for d, m , g in itertools.product(*[domains, methods, graph_names]):
        try:
            df = pd.read_hdf(path+
                "%s_%s_%s_lp_%s_data_hyp.h5" % (d, g, m, s_sch),
                "df"
            )
        except:
            print('%s_%s_%s_lp_%s_data_hyp.h5 not found. Ignoring data set' % (d, g, m, s_sch))
            continue
        df["Domain"], df["Method"], df["Graph"] = d, m, g
        df_all = df_all.append(df).reset_index()
        df_all = df_all.drop(['index'], axis=1)
    if df_all.empty:
        return
    df_all = df_all.drop(['dia'], axis=1)
    plot_shape = (len(domains), len(graph_attrs))
    fin1, axarray1 = plt.subplots(len(domains), len(graph_attrs), figsize=(7, 4), sharex='col', sharey='row')
    data_idx = 0
    gfs_score = {}
    for dom in domains:
        gfs_score[dom] = {m: 0 for m in methods}
        n_attr = 0
        for attr in graph_attrs:
            plot_idx = np.unravel_index(data_idx, plot_shape)
            data_idx += 1
            try:
                rem_attrs = list(set(graph_attrs) - {attr})
                df_grouped = df_all[df_all["Domain"]==dom]
                for rem_attr in rem_attrs:
                    df_grouped = df_grouped[df_grouped[rem_attr]==attr_defaults[rem_attr]]
                df_grouped = df_grouped[[attr, "Round Id", "LP MAP", "Method", "Graph"]]
                print(df_grouped.head())
                df_grouped['LP MAP'] = df_grouped['LP MAP'].astype('float')
                df_grouped = df_grouped.groupby([attr, "Round Id", "Method", "Graph"]).mean().reset_index()
            except TypeError:
                df_trun[hyp_key_ren + "2"] = \
                    df_trun[hyp_key_ren].apply(lambda x: str(x))
                df_trun[hyp_key_ren] = df_trun[hyp_key_ren + "2"].copy()
                df_trun = df_trun.drop([hyp_key_ren + "2"], axis=1)
                df_grouped = df_trun.groupby([hyp_key_ren, "Round Id", "Data"]).max().reset_index()  

            df_grouped['unit']=df_grouped.apply(lambda x:'%s_%s' % (x['Round Id'],x['Graph']),axis=1)
            df_grouped = df_grouped.drop(['Round Id', "Graph"], axis=1)
            m_scores = dict(df_grouped.groupby(["Method"])["LP MAP"].mean())
            n_attr += 1
            gfs_score[dom] = {m: m_scores[m] + gfs_score[dom][m] for m in methods}
            ax = seaborn.tsplot(time=attr, value="LP MAP",
                                    unit="unit", condition="Method", legend=False,
                                    data=df_grouped, ax=axarray1[plot_idx[0], plot_idx[1]])
            if not plot_idx[1]:
                ax.set_ylabel(dom)
            if plot_idx[1]:
                ax.set_ylabel('')
            if plot_idx[0] < len(domains) - 1:
                ax.set_xlabel('')
            attr_values = df_grouped[attr].unique()
            l_diff = attr_values[-1] - attr_values[-2]
            f_diff = attr_values[1] - attr_values[0]
            l_f_diff_r = l_diff / f_diff
            if l_f_diff_r > 1:
                log_base = pow(l_f_diff_r, 1.0 / (len(attr_values) - 2))
                ax.set_xscale('log', basex=round(log_base))
            marker = ["o", "s", "D", "^", "v", "8", "*", "p", "1", "h"]
            for line_i in range(len(ax.lines)):
                ax.lines[line_i].set_marker(marker[line_i])
        gfs_score[dom] = {m: gfs_score[dom][m]/n_attr for m in methods}
    print(gfs_score)
    plt.savefig(
       'benchmark.pdf', 
       dpi=300, format='pdf', bbox_inches='tight'
    )
    plt.show()

def plot_benchmark_individual(domains, graph_attrs, attr_defaults, methods, graph_names, graph_short, s_sch='rw'):
    df_all = pd.DataFrame()
    for d, m , g in itertools.product(*[domains, methods, graph_names]):
        try:
            df = pd.read_hdf(path+
                "%s_%s_%s_lp_%s_data_hyp.h5" % (d, g, m, s_sch),
                "df"
            )
        except:
            print('%s_%s_%s_lp_%s_data_hyp.h5 not found. Ignoring data set' % (d, g, m, s_sch))
            continue
        df["Method"], df["Graph"] = m, g
        df_all = df_all.append(df).reset_index()
        df_all = df_all.drop(['index'], axis=1)
    if df_all.empty:
        return
    df_all = df_all.drop(['dia'], axis=1)
    plot_shape = (len(graph_names), len(graph_attrs))
    fin1, axarray1 = plt.subplots(len(graph_names), len(graph_attrs), figsize=(7, 4), sharex='col', sharey='row')
    data_idx = 0
    for graph in graph_names:
        for attr in graph_attrs:
            plot_idx = np.unravel_index(data_idx, plot_shape)
            data_idx += 1
            try:
                rem_attrs = list(set(graph_attrs) - {attr})
                df_grouped = df_all[df_all["Graph"]==graph]
                for rem_attr in rem_attrs:
                    df_grouped = df_grouped[df_grouped[rem_attr]==attr_defaults[rem_attr]]
                df_grouped = df_grouped[[attr, "Round Id", "LP MAP", "Method"]]
                print(df_grouped.head())
                df_grouped['LP MAP'] = df_grouped['LP MAP'].astype('float')
                df_grouped = df_grouped.groupby([attr, "Round Id", "Method"]).mean().reset_index()
            except TypeError:
                df_trun[hyp_key_ren + "2"] = \
                    df_trun[hyp_key_ren].apply(lambda x: str(x))
                df_trun[hyp_key_ren] = df_trun[hyp_key_ren + "2"].copy()
                df_trun = df_trun.drop([hyp_key_ren + "2"], axis=1)
                df_grouped = df_trun.groupby([hyp_key_ren, "Round Id", "Data"]).max().reset_index()  

            if data_idx == len(graph_names) * len(graph_attrs):
                legend = True
            else:
                legend = False
            ax = seaborn.tsplot(time=attr, value="LP MAP",
                                    unit="Round Id", condition="Method", legend=legend,
                                    data=df_grouped, ax=axarray1[plot_idx[0], plot_idx[1]])
            if legend:
                ax.legend_.remove()
            if not plot_idx[1]:
                ax.set_ylabel(graph_short[graph])
            if plot_idx[1]:
                ax.set_ylabel('')
            if plot_idx[0] < len(graph_names) - 1:
                ax.set_xlabel('')
            attr_values = df_grouped[attr].unique()
            l_diff = attr_values[-1] - attr_values[-2]
            f_diff = attr_values[1] - attr_values[0]
            l_f_diff_r = l_diff / f_diff
            if l_f_diff_r > 1:
                log_base = pow(l_f_diff_r, 1.0 / (len(attr_values) - 2))
                ax.set_xscale('log', basex=round(log_base))
            marker = ["o", "s", "D", "^", "v", "8", "*", "p", "1", "h"]
            for line_i in range(len(ax.lines)):
                ax.lines[line_i].set_marker(marker[line_i])
    for col_idx in range(axarray1[3].shape[0]):
        box = axarray1[3][col_idx].get_position()
        axarray1[3][col_idx].set_position(
            [box.x0,
             box.y0 + box.height * 0.1,
             box.width,
             box.height * 0.9]
        )
    fin1.legend(loc='lower center', bbox_to_anchor=(0.45, -0.01),
                ncol=len(methods), fancybox=True, shadow=True)
    fin1.savefig(
       'benchmark_individual3.pdf', 
       dpi=300, format='pdf', bbox_inches='tight'
    )
    plt.show()


#Path for the stored .h5 result 
path = '/Users/Bench_files/new_files/'



plot_benchmark(
    ["social", "biology", "internet"],
    ["N", "dim", "deg"],
    {"N": 4096, "deg": 8, "dia": None, "dim": 128},
    ["gf", "rand", "pa", "lap", "hope", "cn", "aa","sdne"],
    ["watts_strogatz_graph", "barabasi_albert_graph", "powerlaw_cluster_graph", "random_geometric_graph",\
     "duplication_divergence_graph","hyperbolic_graph","r_mat_graph","waxman_graph","stochastic_block_model",\
     "stochastic_kronecker_graph"],
    s_sch='rw'
)

plot_benchmark_individual(
    ["social", "biology", "internet"],
    ["N", "dim", "deg"],
    {"N": 4096, "deg": 8, "dia": None, "dim": 128},
    ["gf", "rand", "pa", "lap", "hope", "cn", "aa", "sdne"],
    ["watts_strogatz_graph", "barabasi_albert_graph", "powerlaw_cluster_graph", "random_geometric_graph",\
     "duplication_divergence_graph","hyperbolic_graph","r_mat_graph","waxman_graph","stochastic_kronecker_graph"],
    {"watts_strogatz_graph": "WS", "barabasi_albert_graph": "BA", \
     "powerlaw_cluster_graph": "PC", "random_geometric_graph": "RG",\
     "duplication_divergence_graph": "DD", "hyperbolic_graph": "HB",\
     "r_mat_graph": "RM", "waxman_graph":"WM","stochastic_block_model":"SBM", "stochastic_kronecker_graph": "KG"
    },
    s_sch='rw'
)