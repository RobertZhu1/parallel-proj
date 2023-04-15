# Parallel-Project

CSCI 4320 Parallel Programming Project

## About

Our project is a simulation of flights travelling within a network of airports. We structured our
simulation to split up the flights and airports based on the number of ranks in order to achieve a high
degree of parallelism. Within each rank, we structured our simulation to take advantage of the number of GPU
threads available to us by writing a kernel function run by the GPU that handles the main computational work.
After each timeslice, each rank sends flights that belong to another airport to the correct rank that owns that
airport so that the flight can land at the other airport.

## Usage

The generate.py script handles the generation of flights. A number of arguments must be passed into it to obtain a
file usable by the parallel code.

python3 generate.py 1000 120 24 100 flight_test

For example, the above command generates a file named "flight_test" that contains 1000 flights with 120 airports as 
takeoff and landing locations. The specified time period of the flights is 24 hours. "100" represents the seed used for 
randomization.

Next, the code needs to be compiled. Simply running the command "make" will compile the code.

To run the simulation, the following command can be run (though this may vary based on the system used), which utilizes 6 GPUs.
Our tests were performed on AiMOS's DCS cluster.

sbatch -N 1 --ntasks-per-node=6 --gres=gpu:6 -t 30 ./slurmSpectrum.sh

The slurmSpectrum is configured with the arguments to process the file recently generated (300 flights, 20 airports, filename flight_test). 
The flights argument passed in is the number of flights generated divided by the number of ranks, plus added leeway since ranks
may end up with more flights at the end of the simulation. The airports is simply the number of generated airports divided by the number of ranks.

Note: The path in the slurmSpectrum.sh must be changed in order to be used. 