#!/bin/bash -x
module load spectrum-mpi cuda/11.2

taskset -c 0-159:4 mpirun -N 6 /gpfs/u/home/PCPC/PCPCnsss/scratch/Project/sim-cuda-mpi-exe deliver_test 262144 2048 h