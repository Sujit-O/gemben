#!/bin/bash
#SBATCH --ntasks=16
#SBATCH --time=23:59:59


#SBATCH --mail-type=ALL
#SBATCH --mail-user=dh_599@usc.edu
#SBATCH --job-name=benchmark

#SBATCH --error /home/rcf-proj/ef/dh_599/Benchmark/Error/ben_%j.er

source /usr/usc/python/anaconda3/setup.sh
source /usr/usc/cmake/3.7.2/setup.sh


cd /home/rcf-proj/ef/dh_599/Benchmark/GEM-benchmark
KERAS_BACKEND=theano python3 gem/experiments/exp_benchmark2.py -meth ${METH} -domain ${DOM} -rounds 5
