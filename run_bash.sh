#!/bin/bash
for DOM in nxit; do
    for METH in gf; do
        sbatch --export=METH=$METH,DOM=$DOM exp_16core_benchmark.slurm
    done
done
