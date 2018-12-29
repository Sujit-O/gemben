import os
from time import time
from subprocess import call
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import networkx as nx
import scipy

import sys
sys.path.append('./')
sys.path.append(os.path.realpath(__file__))

from gem.utils import graph_util




def barbell_graph(m1,m2):
    graph = nx.barbell_graph(m1,m2)
    ## for com_nc, one hot 
    #onehot_com = np.array([[1,0,0]]*m1+[[0,1,0]]*m2+[[0,0,1]]*m1)  is slower when num of nodes > 2000
    node_labels_com = np.zeros(m1*2+m2).astype(int)
    node_labels_com[m1:m1+m2] = 2
    node_labels_com[m1+m2:] = 1
    ## one hot
    onehot_com = np.zeros((m1*2+m2,3)).astype(int)
    onehot_com[np.arange(m1*2+m2), node_labels_com] = 1
    
    ## for role_nc, one hot
    node_labels_role = np.zeros(m1*2+m2).astype(int)
    p,q = divmod(m2, 2) 
    for i in range(p+1):
        node_labels_role[[m1-1+i,m1+m2-i]] = i+1
    if q:
        node_labels_role[m1+p] = p+2
    onehot_role = np.zeros((m1*2+m2,p+q+2)).astype(int)
    onehot_role[np.arange(m1*2+m2), node_labels_role] = 1

    return graph, scipy.sparse.csr_matrix(onehot_com), scipy.sparse.csr_matrix(onehot_role)

    
    





def binary_community_graph(N, k, maxk, mu):
    ## OS system is windows 
    if sys.platform[0] == "w":
        args = ["gem/c_exe/benchm.exe"]
        fcall = "gem/c_exe/benchm.exe"
    else:
        args = ["gem/c_exe/benchm"]
        fcall = "gem/c_exe/benchm"
    args.append("-N %d" % N)
    args.append("-k %d" % k)
    args.append("-maxk %d" % maxk)
    args.append("-mu %f" % mu)
    t1 = time()
    print(args)
    try:
        os.system("%s -N %d -k %d -maxk %d -mu %f" % (fcall, N, k, maxk, mu))
        # call(args)
    except Exception as e:
        print('ERROR: %s' % str(e))
        print('gem/c_exe/benchm not found. Please compile gf, place benchm in the path and grant executable permission')
    t2 = time()
    print('\tTime taken to generate random graph: %f sec' % (t2 - t1))
    try:
        graph = graph_util.loadGraphFromEdgeListTxt('gem/c_exe/network.dat')
        node_labels = np.loadtxt('gem/c_exe/community.dat')
    except:
        graph = graph_util.loadGraphFromEdgeListTxt('network.dat')
        node_labels = np.loadtxt('community.dat')
    node_labels = node_labels[:, -1].reshape(-1, 1)
    enc = OneHotEncoder()
    return graph, enc.fit_transform(node_labels)
