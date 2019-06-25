try: import cPickle as pickle
except: import pickle
from os import environ
# if 'DISPLAY' not in environ:
#     import matplotlib
#     matplotlib.use('Agg')

import matplotlib
matplotlib.use('Agg')
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


df = pd.read_hdf('graphPropertiesPy3.h5', 'df')
graph_names = [0, 1, 20, 29, 35, 184, 186, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 216, 217, 218, 219, 220, 221, 222, 224, 225, 230, 231, 232, 233, 234, 235, 236, 237, 238, 240, 241, 242, 243, 244, 245, 246, 250, 322, 324, 359, 378, 379, 380, 381, 382, 383, 384]
graph_names_new = [177, 178, 183, 208, 209, 210, 211, 212, 213, 214, 215, 223, 226, 227, 228, 229, 249, 251, 252, 253, 254, 289, 296]
graph_names += graph_names_new
# pdb.set_trace()
df = df[df['Unnamed: 0'].isin(graph_names)]
df['# of nodes'] = df['number_nodes']
df['Avg. degree'] = df['ave_degree']


domains = ['Biological', 'Technological', 'Economic']


# fin1, axarray1 = plt.subplots(3, 2, figsize=(7, 4), sharex='col', sharey='row')
fin1, axarray1 = plt.subplots(len(domains), 2, figsize=(7, 4), sharex='col')
# fin1, axarray1 = plt.subplots(3, 2, figsize=(7, 4))

for idx, domain in enumerate(domains):
    df2 = df[df['networkDomain'] == domain]
    ax = seaborn.distplot(df2['# of nodes'], kde=True, rug=True, ax=axarray1[idx, 0])
    ax.set_xlim([200, 1500])
    ax.set_ylabel(domain)
    if idx < len(domains) - 1:
        ax.set_xlabel('')
    ax = seaborn.distplot(df2['Avg. degree'], kde=True, rug=True, ax=axarray1[idx, 1])

    ax.set_xlim([2, 6])
    ax.set_ylabel('')
    if idx < len(domains) - 1:
        ax.set_xlabel('')
# pdb.set_trace()
plt.savefig(
       'realgraphProps.pdf', # gem/plots/hyp/
       dpi=300, format='pdf', bbox_inches='tight'
    )
# plt.show()
