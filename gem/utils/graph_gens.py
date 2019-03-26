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
import math

def truncate(f, n):
    return math.floor(f * 10 ** n) / 10 ** n
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
def barabasi_albert_graph(N, deg, dia, dim):
    '''
    Parameters of the graph:
    n: Number of Nodes
    m: Number of edges to attach from a new node to existing nodes
    Formula for m:  (m^2)- (Nm)/2 + avg_deg * (N/2) = 0  =>  From this equation we need to find m :
    :return: Graph Object
    '''

    ## Calculating thof nodes: 10\nNumber of edges: 16\nAverage degree:   3.2000'

    if dia is not None:
        return None
    strt_time = time()

    m = int(round((N - np.sqrt(N**2 - 4*deg*N))/4))

    best_G = nx.barabasi_albert_graph(n=N, m=m)

    best_diam = nx.algorithms.diameter(best_G)
    best_avg_deg = np.mean(list(dict(nx.degree(best_G)).values()))


    end_time = time()

    print('Graph_Name: barabase_albert_graph')
    print('Num_Nodes: ', nx.number_of_nodes(best_G), ' Avg_Deg : ', best_avg_deg, ' Diameter: ', best_diam)
    print('TIME: ' , end_time - strt_time, ' secs')

    return best_G

########################################################################################################################

def random_geometric_graph(N, deg, dia, dim):
    '''
    Parameters of the graph:
    n (int or iterable) – Number of nodes or iterable of nodes

    radius (float) – Distance threshold value

    Average Degree is given by formula: Avg_Deg = (pi*(r^2)*num_nodes)/(l^2)
    Formula for r: avg_deg * l
    where l can be considered a constant where its square can be approximated to 1.04 [ength of square] Empirically Found
    :return: Graph Object
    '''
    strt_time = time()

    l = 1.04

    count = 0
    tolerance = 0.3
    curr_deg_error = float('inf')

    while tolerance < curr_deg_error:
        r = np.round(np.sqrt((deg * l ) / (3.14 * N)), 3)

        G = nx.random_geometric_graph(n=N, radius=r)

        curr_avg_deg = np.mean(list(dict(nx.degree(G)).values()))

        lcc = graph_util.get_lcc(G.to_directed())[0]

        curr_diam = nx.algorithms.diameter(lcc)

        curr_deg_error = curr_avg_deg - deg
        count += 1

        if count == 1000:

            break

    best_G = G
    best_diam = curr_diam
    best_avg_deg = curr_avg_deg

    end_time = time()

    print('Graph_Name: Random_Geometric_Graph')
    print('Num_Nodes: ', nx.number_of_nodes(best_G), ' Avg_Deg : ', best_avg_deg, ' Diameter: ', best_diam)
    print('TIME: ', end_time - strt_time)
    return best_G, best_avg_deg, best_diam

########################################################################################################################

def waxman_graph(N, deg, dia, dim):
    '''
    Parameters of the graph:
    n (int or iterable) – Number of nodes or iterable of nodes

    beta (float) – Model parameter

    alpha (float) – Model parameter


    Average Degree is given by formula: Avg_Deg = (n-1) * P
    where P = beta * exp(-d/alpha*L)
    So we fix the parameter beta = 0.1, and we know the default value of d/L is in range: 0.25 to 0.3 (Empiricially calculated)
    so we only tweak alpha to get the required avg deg.

    :return: Graph Object
    '''
    strt_time = time()

    bands = 5
    lower_lim = 0.25
    upper_lim = 0.3
    tolerance = 0.3

    d_by_L_space = np.linspace(lower_lim, upper_lim, bands)
    beta = 0.1
    avg_deg_error_list = []

    for d_by_L in d_by_L_space:
        d_by_L = truncate(d_by_L,4)

        alpha = truncate((-1*d_by_L) / np.log(deg / ((N - 1) * beta)), 2)

        G = nx.waxman_graph(n=N, alpha=alpha, beta=beta)

        lcc = graph_util.get_lcc(G.to_directed())[0]

        curr_avg_deg = np.mean(list(dict(nx.degree(G)).values()))


        curr_diam = nx.algorithms.diameter(lcc)

        avg_deg_error_list.append((G, abs(curr_avg_deg - deg), curr_avg_deg, curr_diam))


    sorted_avg_deg_err = sorted(avg_deg_error_list, key=lambda x: x[1])

    best_G = sorted_avg_deg_err[0][0]

    best_avg_deg= sorted_avg_deg_err[0][2]

    best_diam = sorted_avg_deg_err[0][3]

    end_time = time()

    print('Graph_Name: waxman_graph')
    print('Num_Nodes: ', nx.number_of_nodes(best_G), ' Avg_Deg : ', best_avg_deg, ' Diameter: ', best_diam)
    print('TIME: ', end_time - strt_time)
    return best_G, best_avg_deg, best_diam

########################################################################
def watts_strogatz_graph(N, deg, dia, dim):
    '''
    Parameters of the graph:
    n (int) – The number of nodes
    k (int) – Each node is joined with its k nearest neighbors in a ring topology.
    p (float) – The probability of rewiring each edge

    Average Degree is solely decided by k
    Diameter depends on the value of p
    :return: Graph Object
    '''
    strt_time = time()

    p = 0.2

    best_G = nx.watts_strogatz_graph(n=N, k=deg, p=p)
    best_diam = nx.algorithms.diameter(best_G)
    best_avg_deg = np.mean(list(dict(nx.degree(best_G)).values()))
    end_time = time()

    print('Graph_Name: Watts_Strogatz_Graph')
    print('Num_Nodes: ', nx.number_of_nodes(best_G), ' Avg_Deg : ', best_avg_deg, ' Diameter: ', best_diam)
    print('TIME: ', end_time - strt_time)
    return best_G, best_avg_deg, best_diam

########################################################################
def duplication_divergence_graph(N, deg, dia, dim):
    '''
    Parameters of the graph:
    n (int) – The desired number of nodes in the graph.
    p (float) – The probability for retaining the edge of the replicated node.
    :return: Graph Object
    '''
    strt_time = time()

    tolerance = 0.3
    lower_lim = 0.001
    upper_lim = 1
    bands = 10


    curr_avg_deg_error = float('inf')

    while curr_avg_deg_error > tolerance:
        p_space = np.linspace(lower_lim, upper_lim, bands)
        avg_deg_err_list = []

        p_gap = p_space[1] - p_space[0]
        for p_val in p_space:
            G = nx.duplication_divergence_graph(n=N, p=p_val)

            curr_avg_deg = np.mean(list(dict(nx.degree(G)).values()))

            curr_avg_deg_error = abs(deg - curr_avg_deg)

            avg_deg_err_list.append((G, curr_avg_deg_error, p_val))

        sorted_avg_err = sorted(avg_deg_err_list, key=lambda x: x[1])

        curr_avg_deg_error = sorted_avg_err[0][1]
        if sorted_avg_err[0][1] <= tolerance:

            best_G = sorted_avg_err[0][0]
            break
        else:
            lower_lim = sorted_avg_err[0][2] - p_gap
            upper_lim = sorted_avg_err[0][2] + p_gap

    # best_G = nx.duplication_divergence_graph(n=num_nodes, p=best_p)

    best_diam = nx.algorithms.diameter(best_G)
    best_avg_deg = np.mean(list(dict(nx.degree(best_G)).values()))

    end_time = time()
    print('Graph_Name: powerlaw_cluster_graph')
    print('Num_Nodes: ', nx.number_of_nodes(best_G), ' Avg_Deg : ', best_avg_deg, ' Diameter: ', best_diam)
    print('TIME: ', end_time - strt_time)
    return best_G, best_avg_deg, best_diam

########################################################################
def powerlaw_cluster_graph(N, deg, dia, dim):
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

    m = int(round((N - np.sqrt(N ** 2 - 4 * deg * N)) / 4))
    p = 0.2

    ## G at center:
    best_G = nx.powerlaw_cluster_graph(n=N, m=m, p=p)
    best_diam = nx.algorithms.diameter(best_G)
    best_avg_deg = np.mean(list(dict(nx.degree(best_G)).values()))


    end_time = time()
    print('Graph_Name: powerlaw_cluster_graph')
    print('Num_Nodes: ', nx.number_of_nodes(best_G), ' Avg_Deg : ', best_avg_deg, ' Diameter: ', best_diam)
    print('TIME: ', end_time - strt_time)
    return best_G, best_avg_deg, best_diam






#####################################################################
if __name__=='__main__':

    G= barabasi_albert_graph(1024, 8, None, 128)
