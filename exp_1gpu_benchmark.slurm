#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:59

#SBATCH --mail-type=ALL
#SBATCH --mail-user=palashgo@usc.edu
#SBATCH --job-name=benchmark

cd /home/rcf-proj/ef/palashgo/Bench-2.3/GEM-benchmark
unset PYTHONPATH
source /usr/usc/python/3.6.0/setup.sh
KERAS_BACKEND=theano python3 gem/experiments/exp_benchmark2.py -meth ${METH} -graph ${GRAPH} -rounds 5
