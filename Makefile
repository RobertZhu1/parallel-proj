all: sim-main.c sim-with-cuda.cu
	mpicc -g sim-main.c -c -o sim-main.o
	nvcc -g -G -arch=sm_70 sim-with-cuda.cu -c -o sim-cuda.o
	mpicc -g sim-main.o sim-cuda.o -o sim-cuda-mpi-exe -L/usr/local/cuda-11.2/lib64/ -lcudadevrt -lcudart -lstdc++
