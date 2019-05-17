#!/bin/bash
for GRAPH in random_geometric_graph waxman_graph stochastic_block_model barabasi_albert_graph powerlaw_cluster_graph r_mat_graph watts_strogatz_graph duplication_divergence_graph hyperbolic_graph; do
    for METH in sdne; do
        sbatch --export=METH=$METH,GRAPH=$GRAPH exp_1gpu_benchmark.slurm
    done
done
