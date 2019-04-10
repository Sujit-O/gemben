#!/bin/bash
for DOM in internet ; do
    for METH in lap; do
        sbatch --export=METH=$METH,DOM=$DOM exp_16core_benchmark.slurm
    done
done
