#!/bin/bash
for DOM in social internet biology; do
    for METH in hope; do
        sbatch --export=METH=$METH,DOM=$DOM exp_16core_benchmark.slurm
    done
done
