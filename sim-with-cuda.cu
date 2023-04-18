
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include "airport.h"

extern struct airport *airports;
extern int num_airports;

extern struct flight *flights;
extern int num_flights;

extern struct flight **g_send_flights;
extern struct flight *g_recv_flights; 

// Rank number of current MPI rank
extern int myrank;

// Total number of MPI ranks
extern int numranks;

extern "C" void sim_initMaster()
{
	// Initializes CUDA
	int cE, cudaDeviceCount;
	if( (cE = cudaGetDeviceCount(&cudaDeviceCount)) != cudaSuccess)
	{
		printf("Unable to determine cuda device count, error is %d, count is %d\n", cE, cudaDeviceCount);
		exit(-1);
	}
	if( (cE = cudaSetDevice(myrank % cudaDeviceCount)) != cudaSuccess)
	{
		printf("Unable to have rank %d set to cuda device %d, error is %d\n", myrank, (myrank % cudaDeviceCount), cE);
		exit(-1);
	}
	int recv_size = 10000;
	int send_size = 5000;
	int max_flights = 6000000;
	int max_airports = 361440;

	// Allocates data for the send and recv arrays
	cudaMallocManaged(&g_send_flights, (numranks * sizeof(struct flight *)));
	cudaMallocManaged(&g_recv_flights, recv_size * sizeof(struct flight));
	for(int i = 0; i < numranks; i++){
		cudaMallocManaged(&g_send_flights[i], send_size * sizeof(struct flight));
	}

	// Allocates space for the flights and airports
	cudaMallocManaged(&flights, max_flights * sizeof(struct flight));
	cudaMallocManaged(&airports, max_airports * sizeof(struct airport));

	// Initialize airports and flights
	for(int i = 0; i < num_airports; i++){
		airports[i].num_runways = 1;
		cudaMallocManaged(&airports[i].runways, airports[i].num_runways * sizeof(struct runway));
	}
	for(int i = 0; i < num_flights; i++){
		flights[i].stage = INVALID_FLIGHT;
	}

}

/* sim_kernel is the kernel function called from the host. 
   It simulates one unit of discrete time.
 */
__global__ void sim_kernel(struct flight * flights,
                           struct airport * airports, 
                           unsigned int num_flights,
                           unsigned int num_airports,
                           struct flight * recv_flights,
                           struct flight ** send_flights,
                           int * count_sent,
                           int numranks,
                           unsigned int current_time){
    /*
	PSUDO CODE:
	-> calculate thread id
	-> go to that spot in the array, and get the airplane that is going to land next
	-> see where it is supposed to land
	-> try to change the airport that it will land at by adding it to the runway queue 
	*/

	int index;
	
	// Loops through flights and computes their next stage
	for(index = blockIdx.x * blockDim.x + threadIdx.x; index < num_flights; index += blockDim.x * gridDim.x)
	{
		// Checks recv flights to see if any new flights can be added in
		if(flights[index].stage == INVALID_FLIGHT){
			int recv_index = atomicSub(&count_sent[numranks], 1) - 1;
			if(recv_index >= 0){
				memcpy(&flights[index], &recv_flights[recv_index], sizeof(struct flight));
			}
		}

		int flight_stage = flights[index].stage;

		// Sets flight stage if flight has reached takeoff time
		if(flight_stage == WAITING && flights[index].starting_time <= current_time){
			flights[index].stage = PREPARING_TO_TAKEOFF;
			flight_stage = PREPARING_TO_TAKEOFF;
		}

		if(flight_stage == PREPARING_TO_TAKEOFF){
			int airport_index = flights[index].source_id / numranks;

			// Attempts to obtain a runway in order to takeoff
			for(int i = 0; i < airports[airport_index].num_runways; i++){
				int available = !atomicCAS(&airports[airport_index].runways[i].status, 0, 1);
				if(available){
					if(airports[airport_index].runways[i].last_access != current_time){
						flights[index].stage = TAKEOFF;
						flights[index].current_runway_id = i;
						flights[index].taxi_time = 1;
						flights[index].taxi_time--;
						break;
					}
					else{
						atomicExch(&airports[airport_index].runways[i].status, 0);
					}
				}
			}

			// Increments wait time if flight could not get runway
			if(flights[index].stage == PREPARING_TO_TAKEOFF){
				flights[index].wait_time++;
			}
		}
		else if(flight_stage == TAKEOFF){
			flights[index].taxi_time--;

			// Advances the stage to cruising if the flight has completed taxiing
			if(flights[index].taxi_time == 0){
				flights[index].stage = CRUISING;
				int airport_index = flights[index].source_id / numranks;
				int runway_index = flights[index].current_runway_id;

				// Sets last access time of runway and releases runway
				airports[airport_index].runways[runway_index].last_access = current_time;
				atomicExch(&airports[airport_index].runways[runway_index].status, 0);
			}
		}
		else if(flight_stage == CRUISING){
			flights[index].travel_time--;
			if(flights[index].travel_time == 0){
				flights[index].stage = PREPARING_TO_LAND;

				// Gets the appropriate rank to send to
				int current_rank = flights[index].source_id % numranks;
				int dest_rank = flights[index].destination_id % numranks;

				// Gets the index to write into and increments the number of elements
				int send_index = atomicAdd(&count_sent[dest_rank], 1);
				if(current_rank != dest_rank){
					memcpy(&send_flights[dest_rank][send_index], &flights[index], sizeof(struct flight));
					flights[index].stage = INVALID_FLIGHT;
				}
			}
		}
		else if(flight_stage == PREPARING_TO_LAND){
			int airport_index = flights[index].destination_id / numranks;

			// Attempts to obtain a runway in order to land
			for(int i = 0; i < airports[airport_index].num_runways; i++){
				int available = !atomicCAS(&airports[airport_index].runways[i].status, 0, 1);
				if(available){
					if(airports[airport_index].runways[i].last_access != current_time){
						flights[index].stage = LANDING;
						flights[index].current_runway_id = i;
						flights[index].taxi_time = 1;
						flights[index].taxi_time--;
						break;
					}
					else{
						atomicExch(&airports[airport_index].runways[i].status, 0);
					}
				}
			}

			// Increments wait time if flight could not get runway
			if(flights[index].stage == PREPARING_TO_LAND){
				flights[index].wait_time++;
			}
		}
		else if(flight_stage == LANDING){
			flights[index].taxi_time--;
			if(flights[index].taxi_time == 0){
				flights[index].stage = LANDED;
				flights[index].landing_time = current_time + 1;
				int airport_index = flights[index].destination_id / numranks;
				int runway_index = flights[index].current_runway_id;

				// Sets the last accessed time of the runway and releases it
				airports[airport_index].runways[runway_index].last_access = current_time;
				atomicExch(&airports[airport_index].runways[runway_index].status, 0);
			}
		}

	} 
}

// printFlight is a helper function that prints out data of the passed in flight
extern "C" void printFlight(struct flight f){
	printf("Flight id %d:\n", f.id);
	if(f.stage == INVALID_FLIGHT){
		printf("INVALID FLIGHT\n");
		return;
	}

	printf("Source: %d\nDest: %d\n", f.source_id, f.destination_id);
	if(f.stage == WAITING){
		printf("Stage: WAITING\n");
	}
	else if(f.stage == PREPARING_TO_TAKEOFF){
		printf("Stage: PREPARING TO TAKEOFF\n");
	}
	else if(f.stage == TAKEOFF){
		printf("Stage: TAKEOFF\n");
	}
	else if(f.stage == CRUISING){
		printf("Stage: CRUISING\n");
	}
	else if(f.stage == PREPARING_TO_LAND){
		printf("Stage: PREPARING TO LAND\n");
	}
	else if(f.stage == LANDING){
		printf("Stage: LANDING\n");
	}
	else if(f.stage == LANDED){
		printf("Stage: LANDED\n");
	}
	else{
		printf("INVALID STAGE!!!\n");
	}

	printf("Runway id: %d\n", f.current_runway_id);
	printf("Travel time: %d\n", f.travel_time);
	printf("Taxi time: %d\n", f.taxi_time);
	printf("Wait time: %d\n", f.wait_time);
}

/* sim_kernelLaunch launches the kernel from the host.
 */
extern "C" bool sim_kernelLaunch(int * count_to_send, unsigned int current_time)
{
	sim_kernel<<<32,32>>>(flights, airports, num_flights, num_airports, g_recv_flights, g_send_flights, count_to_send, numranks, current_time);
	//sim_kernel<<<1,1>>>(flights, airports, num_flights, num_airports, g_recv_flights, g_send_flights, count_to_send, numranks, current_time);

    // Calls this to guarantee that the kernel will finish computation before swapping
    cudaDeviceSynchronize();
    
    return true;    
}


