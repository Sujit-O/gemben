#!/bin/bash
for DOM in social internet biology; do
    for METH in rand aa gf hope; do
        sbatch --export=METH=$METH,DOM=$DOM exp_16core_benchmark.slurm
    done
    sbatch --export=METH=sdne,DOM=$DOM exp_1gpu_benchmark.slurm
done
