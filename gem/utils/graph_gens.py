#!/usr/local/bin/python
# coding: utf-8
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


########################################################################
def barabasi_albert_graph(num_nodes, avg_deg, diam, emb_dim):
    '''
    Parameters of the graph:
    n: Number of Nodes
    m: Number of edges to attach from a new node to existing nodes
    Formula for m:  (m^2)- (Nm)/2 + avg_deg * (N/2) = 0  =>  From this equation we need to find m :
    :return: Graph Object
    '''

    ## Calculating thof nodes: 10\nNumber of edges: 16\nAverage degree:   3.2000'
    strt_time = time()

    center = avg_deg//2

    m_list = []
    for i in range(1,3):
        m_list.insert(0,center - i)
        m_list.append(center + i)

    ## G at center:
    G = nx.barabasi_albert_graph(n=num_nodes, m=center)
    curr_diam = nx.algorithms.diameter(G)
    curr_avg_deg = np.mean(nx.degree(G).values())

    diam_error = abs(diam - curr_diam)
    avg_deg_error = abs(avg_deg - curr_avg_deg)
    total_error = diam_error + avg_deg_error
    best_G = G
    best_diam = curr_diam
    best_avg_deg = curr_avg_deg

    for m_val in m_list:
        G = nx.barabasi_albert_graph(n = num_nodes, m = m_val)
        curr_diam = nx.algorithms.diameter(G)
        curr_avg_deg = np.mean(nx.degree(G).values())
        curr_diam_error = abs(diam - curr_diam)
        curr_avg_deg_error = abs(avg_deg - curr_avg_deg)

        if curr_diam_error+curr_avg_deg_error < total_error:
            total_error = curr_diam_error + curr_avg_deg_error
            best_G = G
            best_diam = curr_diam
            best_avg_deg = curr_avg_deg

    end_time = time()
    print 'Graph_Name: barabase_albert_graph'
    print 'Num_Nodes: ', nx.number_of_nodes(G), ' Avg_Deg : ', best_avg_deg, ' Diameter: ', best_diam
    print 'TIME: ' , end_time - strt_time
    return best_G



########################################################################
def powerlaw_cluster_graph(num_nodes, avg_deg, diam, emb_dim):
    '''
    Parameters of the graph:
    n (int) – the number of nodes
    m (int) – the number of random edges to add for each new node
    p (float,) – Probability of adding a triangle after adding a random edge
    Formula for m:  (m^2)- (Nm)/2 + avg_deg * (N/2) = 0  =>  From this equation we need to find m :
    p : Does not vary the average degree or diameter so much. : Higher value of p may cause average degree to overshoot intended average_deg
    so we give the control of average degree to parameter m: by setting a lower value of p: 0.1
    :return: Graph Object
    '''

    ## Calculating thof nodes: 10\nNumber of edges: 16\nAverage degree:   3.2000'
    strt_time = time()

    center = avg_deg//2
    default_p = 0.1
    m_list = []
    for i in range(1,3):
        m_list.insert(0,center - i)
        m_list.append(center + i)

    ## G at center:
    G = nx.powerlaw_cluster_graph(n=num_nodes, m=center , p=default_p)
    curr_diam = nx.algorithms.diameter(G)
    curr_avg_deg = np.mean(nx.degree(G).values())

    diam_error = abs(diam - curr_diam)
    avg_deg_error = abs(avg_deg - curr_avg_deg)
    total_error = diam_error + avg_deg_error
    best_G = G
    best_diam = curr_diam
    best_avg_deg = curr_avg_deg

    for m_val in m_list:
        G = nx.powerlaw_cluster_graph(n = num_nodes, m = m_val, p = default_p)
        curr_diam = nx.algorithms.diameter(G)
        curr_avg_deg = np.mean(nx.degree(G).values())
        curr_diam_error = abs(diam - curr_diam)
        curr_avg_deg_error = abs(avg_deg - curr_avg_deg)

        if curr_diam_error+curr_avg_deg_error < total_error:
            total_error = curr_diam_error + curr_avg_deg_error
            best_G = G
            best_diam = curr_diam
            best_avg_deg = curr_avg_deg

    end_time = time()
    print 'Graph_Name: powerlaw_cluster_graph'
    print 'Num_Nodes: ', nx.number_of_nodes(G), ' Avg_Deg : ', best_avg_deg, ' Diameter: ', best_diam
    print 'TIME: ' , end_time - strt_time
    return best_G


########################################################################
def duplication_divergence_graph(num_nodes, avg_deg, diam, emb_dim):
    '''
    Parameters of the graph:
    n (int) – The desired number of nodes in the graph.
    p (float) – The probability for retaining the edge of the replicated node.
    :return: Graph Object
    '''

    ## Calculating thof nodes: 10\nNumber of edges: 16\nAverage degree:   3.2000'
    strt_time = time()

    tolerance =0.1
    lower_lim = 0.001
    upper_lim = 1
    bands = 10

    avg_deg_err_list = []

    best_p= 0.5
    curr_avg_deg_error = 1

    while curr_avg_deg_error <= tolerance:
        p_space = np.linspace(lower_lim, upper_lim, bands)

        p_gap = p_space[1]-p_space[0]
        for p_val in p_space:
            G = nx.duplication_divergence_graph(n=num_nodes, p=p_val)

            curr_avg_deg = np.mean(nx.degree(G).values())

            curr_avg_deg_error = abs(avg_deg - curr_avg_deg)

            avg_deg_err_list.append((p_val,curr_avg_deg_error))

        sorted_avg_err = sorted(avg_deg_err_list,key=lambda x: x[1])
        if sorted_avg_err[0][1] <= tolerance:
            best_p = sorted_avg_err[0][0]
            break
        else:
            lower_lim = sorted_avg_err[0][0] - p_gap
            upper_lim = sorted_avg_err[0][0] + p_gap


    best_G = nx.duplication_divergence_graph(n=num_nodes, p = best_p)

    best_diam = nx.algorithms.diameter(best_G)
    best_avg_deg = np.mean(nx.degree(best_G).values())

    end_time = time()
    print 'Graph_Name: powerlaw_cluster_graph'
    print 'Num_Nodes: ', nx.number_of_nodes(best_G), ' Avg_Deg : ', best_avg_deg, ' Diameter: ', best_diam
    print 'TIME: ', end_time - strt_time
    return best_G


#####################################################################
if __name__=='__main__':

    # print os.getcwd()
    # file_name = os.path.abspath(os.path.join(os.getcwd(), os.pardir))+"/experiments/config/synthetic/lfr_avgDeg.txt"
    # plot_file = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "/plots/lfr_hist"
    # print file_name

    # G= barabasi_albert_graph(1024, 8, 4, 128)
    # print nx.info(G)
    G = duplication_divergence_graph(1024, 8 , 4, 128)
    

    # if os.path.isfile(file_name):
    #     os.remove(file_name)

    # for i in range(1000):
    #     lancichinetti_fortunato_radicchi(1024,8,4)
    #
    # with open(file_name, "r") as fp:
    #     degrees = fp.readlines()
    # avg_deg = []
    # for deg in degrees:
    #     avg_deg.append(round(float(deg.strip('\n')), 2))
    #
    # print avg_deg
    #
    # print plot_file
    # plot_hist(plot_file,avg_deg)
