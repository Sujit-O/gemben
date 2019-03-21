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


###################################################################
def plot_hist(title,data):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.hist(x=data)
    plt.savefig(title+'.png')

########################################################################

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

##########################################################################

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

########################################################################
def lancichinetti_fortunato_radicchi(N,avg_deg, diam):
    if sys.platform[0] == "w":
        args = ["binary_networks/c_exe/benchmark.exe"]
        fcall = "gem/c_exe/benchm.exe"
    else:
        args = ["binary_networks/c_exe/benchmark"]
        fcall = "binary_networks/c_exe/benchmark"

    mu = 0.1
    minc = 0.1
    maxk = 50
    maxc = 50
    args.append("-N %d" % N)
    args.append("-k %d" % avg_deg)
    args.append("-maxk %d" % maxk)
    args.append("-mu %f" % mu)
    args.append("-minc %f" % minc)
    args.append("-maxc %f" % maxc)

    t1 = time()
    try:
        os.chdir('/home/ankita/studies/usc_research/GEM-benchmark/')
        print os.getcwd()
        # print ("%s -N %d -k %d -maxk %d -mu %f -minc %d -maxc %d" % (fcall, N, avg_deg, maxk, mu, minc, maxc))
        os.system("%s -N %d -k %d -maxk %d -mu %f -minc %d -maxc %d" % (fcall, N, avg_deg, maxk, mu, minc, maxc))
        # call(args)
    except Exception as e:
        print('ERROR: %s' % str(e))
        print('binary_networks/c_exe/benchmark not found. Please compile gf, place benchmark in the path and grant executable permission')
    print(args)
    t2 = time()

    print 'time_taken: ', t2-t1, ' secs'


#####################################################################
if __name__=='__main__':

    print os.getcwd()
    file_name = os.path.abspath(os.path.join(os.getcwd(), os.pardir))+"/experiments/config/synthetic/lfr_avgDeg.txt"
    plot_file = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "/plots/lfr_hist"
    print file_name
    if os.path.isfile(file_name):
        os.remove(file_name)

    for i in range(1000):
        lancichinetti_fortunato_radicchi(1024,8,4)

    with open(file_name, "r") as fp:
        degrees = fp.readlines()
    avg_deg = []
    for deg in degrees:
        avg_deg.append(round(float(deg.strip('\n')), 2))

    print avg_deg

    print plot_file
    plot_hist(plot_file,avg_deg)
