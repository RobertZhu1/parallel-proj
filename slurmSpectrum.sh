#!/bin/bash -x

module load gcc/7.4.0/1
module load spectrum-mpi
module load cuda

mpirun -np 1 /gpfs/u/home/PCPC/PCPCzhtn/barn/sim-cuda-mpi-exe flight_test 300 20
