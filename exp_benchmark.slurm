#!/bin/bash
#SBATCH --ntasks=8
#SBATCH --time=23:59:59
cd /home/rcf-proj/ef/palashgo/GEM
KERAS_BACKEND=theano python gem/experiments/exp_benchmark.py -data ${DATA} -meth rand,pa,cn,jc,aa,lap,hope,gf -dim ${DIM}
