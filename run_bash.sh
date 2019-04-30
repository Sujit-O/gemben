#!/bin/bash
for DOM in social internet biology; do
    for METH in gf lap pa hope rand cn aa jc; do
        sbatch --export=METH=$METH,DOM=$DOM exp_16core_benchmark.slurm
    done
done
