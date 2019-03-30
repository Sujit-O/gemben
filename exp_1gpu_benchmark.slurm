#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:59
cd /home/rcf-proj/ef/palashgo/GEM-benchmark
KERAS_BACKEND=theano python gem/experiments/exp_benchmark2.py -meth ${METH} -domain ${DOM} -rounds 5
