# Parallel-Project

CSCI 6490 Parallel Programming Project

## About

For the final team project, we decide to implement a discrete time simulation of logistics using MPI, CUDA, and MPI I/O. We are particularly interested in simulating logistics because each individual delivery event is relatively simple to model while still worth studying as the system scales up to millions of deliveries daily. Also, logistics as a topic is very close to our daily life as online shopping is very popular nowadays. It refers to the management of the flow of goods, services, and information between the point of origin and the point of consumption. It involves the planning, coordination, and execution of activities such as transportation, warehousing, inventory management, and order fulfillment to ensure that products are delivered to customers in a timely, efficient, and cost-effective manner. Logistics is an important function of supply chain management, as it plays a key role in ensuring that products are available to customers when and where they need them. Effective logistics management requires a deep understanding of transportation networks, inventory management practices, and customer demand patterns, among other factors. By optimizing logistics processes, companies can improve their operational efficiency, reduce costs, and enhance customer satisfaction.

More specifically, we would like to simulate logistic delays. Logistics delays are important to simulate because they can have significant impacts on various aspects of a supply chain. Logistics delays can lead to longer lead times, which can impact the delivery of goods to customers, increase inventory holding costs, and reduce customer satisfaction. Delays in logistics can also cause disruptions in the supply chain, resulting in decreased efficiency and increased costs. For example, if a shipment is delayed, it may cause a delay in the production schedule, which can lead to increased labor costs and reduced productivity. Logistics delays can result in increased costs, such as additional transportation costs, increased inventory holding costs, and potential penalties for not meeting delivery deadlines. By simulating logistics delays, supply chain managers can identify potential bottlenecks in the supply chain, evaluate the impact of different scenarios, and make more informed decisions about inventory management, transportation planning, and other critical aspects of the supply chain. This can lead to increased efficiency, reduced costs, and improved customer satisfaction.

We are going to model deliveries throughout one day by processing the deliveries in parallel at different timestamps using MPI ranks and CUDA GPU. We will also use MPI I/O to process input deliveries and output statistics. For the performance analysis, we will perform strong scaling and week scaling using 1, 2, 4, and 8 compute nodes. The throughput of the MPI read and write along with the simulation time would be recorded and compared among the different computer resource configurations.

## Usage

### Generate deliveries
The **generate.py script** randomly generates a list of deliveries given simultation duration.

Execution: python generate.py 1572864 12288 24 100 delivery_test 

This would generate a file named "delivery_test" with 1572864 deliveries within 24 hours. The deliveries would transit in a network of 12288 transit centers. The seed of the random generator would be set to 100.

### Make executables of source code

Execute "make" in the terminal should create executables out of **sim-main.c**, **transit_center.h**, and **sim-with-cuda.cu**.

### Make a run script for the executable

To run the executable we need to pass in the list of deliveries we generated using **generate.py**. We also need to pass in the number of deliveries and number of transit centers hold by each rank. We pass in "h" optionally to specify whether we want to run the executable in hybrid mode or MPI only mode. An example of the run script is shown below:

```bash
#!/bin/bash -x
module load spectrum-mpi cuda/11.2

taskset -c 0-159:4 mpirun -N 6 /gpfs/u/home/PCPC/PCPCnsss/scratch/Project/sim-cuda-mpi-exe deliver_test 262144 2048 h
```

This line would execute the source code with 6 ranks per compute node. Each rank would deal with 262144 deliveries and has access to 2048 compute centers. The executable would run in hybrid mode so 6 GPUs would be utilized.

### Submit the run script as a job to AiMOS

To run the script on AiMOS, we need to specify the computer resource configuration and submit the job to SLURM. An example of SLURM command line to submit the job is shown below:

```bash
sbatch -N 2 --partition=el8 --gres=gpu:6 -t 10 -o /gpfs/u/home/PCPC/PCPCnsss/scratch/Project/tmp.out ./run.sh
```

This line would request for 2 compute nodes with 6 GPUs on each  in e18 partition for 10 minutes. The output of the execution would be written to tmp.out.

Note: The path in the run.sh and slurm.sh should be absolute paths and modified according to where it locates. 