
def plot_hist(title,data):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.hist(x=data)
    plt.savefig(title+'.png')


def get_diams(graph_nme, hyps, num_iters):
    diam_list = []
    avg_deg_list = []
    if graph_nme == 'barabasi_albert_graph':
        n = hyps['n']
        m = hyps['m']
        for i in range(num_iters):
            if i % 10 == 0:







                print graph_nme, '______________ ', i
            G = nx.barabasi_albert_graph(n=n, m=m)
            diam_list.append(nx.algorithms.diameter(G))
            avg_deg_list.append(np.mean(nx.degree(G).values()))

        plot_hist(graph_nme + '_AvgDeg'+'_iter_'+str(num_iters), avg_deg_list)
        plot_hist(graph_nme+'_Diam'+'_iter_'+str(num_iters), diam_list)
        print '=========================== '
        print avg_deg_list

    if graph_nme == 'watts_strogatz_graph':
        n = hyps['n']
        k = hyps['k']
        p = hyps['p']
        for i in range(num_iters):
            if i % 10 == 0:
                print graph_nme, '______________ ', i
            G = nx.watts_strogatz_graph(n=n, k=k,p=p)
            diam_list.append(nx.algorithms.diameter(G))
            avg_deg_list.append(np.mean(nx.degree(G).values()))

        plot_hist(graph_nme + '_AvgDeg'+'_iter_'+str(num_iters), avg_deg_list)
        plot_hist(graph_nme+'_Diam'+'_iter_'+str(num_iters), diam_list)
        print '=========================== '
        print avg_deg_list

    if graph_nme == 'random_geometric_graph':
        n = hyps['n']
        radius = hyps['radius']
        for i in range(num_iters):
            if i % 10 == 0:
                print graph_nme, '______________ ', i
            G = nx.random_geometric_graph(n=n, radius=radius)
            diam_list.append(nx.algorithms.diameter(G))
            avg_deg_list.append(np.mean(nx.degree(G).values()))

        plot_hist(graph_nme + '_AvgDeg' + '_iter_' + str(num_iters), avg_deg_list)
        plot_hist(graph_nme + '_Diam' + '_iter_' + str(num_iters), diam_list)
        print '=========================== '
        print avg_deg_list

    if graph_nme == 'powerlaw_cluster_graph':
        n = hyps['n']
        m = hyps['m']
        p = hyps['p']

        for i in range(num_iters):
            if i%10 == 0:
                print graph_nme, '______________ ',i
            G = nx.powerlaw_cluster_graph(n=n, m=m, p=p)
            diam_list.append(nx.algorithms.diameter(G))
            avg_deg_list.append(np.mean(nx.degree(G).values()))

        plot_hist(graph_nme + '_AvgDeg' + '_iter_' + str(num_iters), avg_deg_list)
        plot_hist(graph_nme + '_Diam' + '_iter_' + str(num_iters), diam_list)
        print '=========================== '
        print avg_deg_list

    if graph_nme == 'duplication_divergence_graph':
        n = hyps['n']
        p = hyps['p']

        for i in range(num_iters):
            if i%10 == 0:
                print graph_nme, '______________ ',i
            G = nx.duplication_divergence_graph(n=n, p=p)
            diam_list.append(nx.algorithms.diameter(G))
            avg_deg_list.append(np.mean(nx.degree(G).values()))

        plot_hist(graph_nme + '_AvgDeg' + '_iter_' + str(num_iters), avg_deg_list)
        plot_hist(graph_nme + '_Diam' + '_iter_' + str(num_iters), diam_list)
        print '=========================== '
        print avg_deg_list

    if graph_nme == 'dorogovtsev_goltsev_mendes_graph':
        n = hyps['n']
        create_using = hyps['create_using']

        for i in range(num_iters):
            if i%10 == 0:
                print graph_nme, '______________ ',i
            G = nx.dorogovtsev_goltsev_mendes_graph(n=n)
            diam_list.append(nx.algorithms.diameter(G))
            avg_deg_list.append(np.mean(nx.degree(G).values()))

        plot_hist(graph_nme + '_AvgDeg' + '_iter_' + str(num_iters), avg_deg_list)
        plot_hist(graph_nme + '_Diam' + '_iter_' + str(num_iters), diam_list)
        print '=========================== '
        print avg_deg_list


if __name__=='__main__':
    import networkx as nx
    import numpy as np
    import json

    with open('diam_syn_hyps.conf','r') as fp:
        syn_hyps = json.load(fp)

    num_iters = 10

    graph_names = syn_hyps.keys()

    for graph_nme in graph_names:

        hyps = syn_hyps[graph_nme]

        get_diams(graph_nme,hyps, num_iters)
