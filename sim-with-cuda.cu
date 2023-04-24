#include<cuda.h>
#include<cuda_runtime.h>
#include "airport.h"

extern struct transit_center *transit_centers;
extern int num_transit_centers;

extern struct delivery *deliveries;
extern int num_deliveries;

extern struct delivery **outgoing_deliveries;
extern struct delivery *incoming_deliveries; 

// Rank number of current MPI rank
extern int world_rank;

// Total number of MPI ranks
extern int world_size;

extern "C" void sim_initMaster()
{
	// Initializes CUDA
	int cE, cudaDeviceCount;
	if( (cE = cudaGetDeviceCount(&cudaDeviceCount)) != cudaSuccess)
	{
		printf("Unable to determine cuda device count, error is %d, count is %d\n", cE, cudaDeviceCount);
		exit(-1);
	}
	if( (cE = cudaSetDevice(world_rank % cudaDeviceCount)) != cudaSuccess)
	{
		printf("Unable to have rank %d set to cuda device %d, error is %d\n", world_rank, (world_rank % cudaDeviceCount), cE);
		exit(-1);
	}
	int recv_size = 10000;
	int send_size = 5000;
	int max_deliveries = 6000000;
	int max_transit_centers = 361440;

	// Allocates data for the send and recv arrays
	cudaMallocManaged(&outgoing_deliveries, (world_size * sizeof(struct delivery *)));
	cudaMallocManaged(&incoming_deliveries, recv_size * sizeof(struct delivery));
	for(int i = 0; i < world_size; i++){
		cudaMallocManaged(&outgoing_deliveries[i], send_size * sizeof(struct delivery));
	}

	// Allocates space for the flights and airports
	cudaMallocManaged(&deliveries, max_deliveries * sizeof(struct delivery));
	cudaMallocManaged(&transit_centers, max_transit_centers * sizeof(struct transit_center));

	// Initialize airports and flights
	for(int i = 0; i < num_transit_centers; i++){
		transit_centers[i].num_conveyor_belts = 1;
		cudaMallocManaged(&transit_centers[i].conveyor_belts, transit_centers[i].num_conveyor_belts * sizeof(struct conveyor_belt));
	}
	for(int i = 0; i < num_deliveries; i++){
		deliveries[i].status = LOST_DELIVERY;
	}

}

/* sim_kernel is the kernel function called from the host. 
   It simulates one unit of discrete time.
 */
__global__ void sim_kernel(struct delivery * deliveries,
                           struct transit_center * transit_centers, 
                           unsigned int num_deliveries,
                           unsigned int num_transit_centers,
                           struct delivery * recv_flights,
                           struct delivery ** send_flights,
                           int * count_sent,
                           int world_size,
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
	for(index = blockIdx.x * blockDim.x + threadIdx.x; index < num_deliveries; index += blockDim.x * gridDim.x)
	{
		// Checks recv flights to see if any new flights can be added in
		if(deliveries[index].status == LOST_DELIVERY){
			int recv_index = atomicSub(&count_sent[world_size], 1) - 1;
			if(recv_index >= 0){
				memcpy(&deliveries[index], &recv_flights[recv_index], sizeof(struct delivery));
			}
		}

		int delivery_status = deliveries[index].status;

		// Sets flight stage if flight has reached takeoff time
		if(delivery_status == WAITING && deliveries[index].starting_time <= current_time){
			deliveries[index].status = STAMPED;
			delivery_status = STAMPED;
		}

		if(delivery_status == STAMPED){
			int transit_center_index = deliveries[index].source_id / world_size;

			// Attempts to obtain a runway in order to takeoff
			for(int i = 0; i < transit_centers[transit_center_index].num_conveyor_belts; i++){
				int available = !atomicCAS(&transit_centers[transit_center_index].conveyor_belts[i].status, 0, 1);
				if(available){
					if(transit_centers[transit_center_index].conveyor_belts[i].last_access != current_time){
						deliveries[index].status = LEAVE_SOURCE;
						deliveries[index].current_conveyor_id = i;
						deliveries[index].processing_time = 1;
						deliveries[index].processing_time--;
						break;
					}
					else{
						atomicExch(&transit_centers[transit_center_index].conveyor_belts[i].status, 0);
					}
				}
			}

			// Increments wait time if flight could not get runway
			if(deliveries[index].status == STAMPED){
				deliveries[index].wait_time++;
			}
		}
		else if(delivery_status == LEAVE_SOURCE){
			deliveries[index].processing_time--;

			// Advances the stage to cruising if the flight has completed taxiing
			if(deliveries[index].processing_time == 0){
				deliveries[index].status = IN_TRANSIT;
				int transit_center_index = deliveries[index].source_id / world_size;
				int conveyor_index = deliveries[index].current_conveyor_id;

				// Sets last access time of runway and releases runway
				transit_centers[transit_center_index].conveyor_belts[conveyor_index].last_access = current_time;
				atomicExch(&transit_centers[transit_center_index].conveyor_belts[conveyor_index].status, 0);
			}
		}
		else if(delivery_status == IN_TRANSIT){
			deliveries[index].transit_time--;
			if(deliveries[index].transit_time == 0){
				deliveries[index].status = ARRIVE_TO_DES;

				// Gets the appropriate rank to send to
				int current_rank = deliveries[index].source_id % world_size;
				int dest_rank = deliveries[index].destination_id % world_size;

				// Gets the index to write into and increments the number of elements
				int send_index = atomicAdd(&count_sent[dest_rank], 1);
				if(current_rank != dest_rank){
					memcpy(&send_flights[dest_rank][send_index], &deliveries[index], sizeof(struct delivery));
					deliveries[index].status = LOST_DELIVERY;
				}
			}
		}
		else if(delivery_status == ARRIVE_TO_DES){
			int transit_center_index = deliveries[index].destination_id / world_size;

			// Attempts to obtain a runway in order to land
			for(int i = 0; i < transit_centers[transit_center_index].num_conveyor_belts; i++){
				int available = !atomicCAS(&transit_centers[transit_center_index].conveyor_belts[i].status, 0, 1);
				if(available){
					if(transit_centers[transit_center_index].conveyor_belts[i].last_access != current_time){
						deliveries[index].status = PROCESSING;
						deliveries[index].current_conveyor_id = i;
						deliveries[index].processing_time = 1;
						deliveries[index].processing_time--;
						break;
					}
					else{
						atomicExch(&transit_centers[transit_center_index].conveyor_belts[i].status, 0);
					}
				}
			}

			// Increments wait time if flight could not get runway
			if(deliveries[index].status == ARRIVE_TO_DES){
				deliveries[index].wait_time++;
			}
		}
		else if(delivery_status == PROCESSING){
			deliveries[index].processing_time--;
			if(deliveries[index].processing_time == 0){
				deliveries[index].status = DELIVERED;
				deliveries[index].arriving_time = current_time + 1;
				int transit_center_index = deliveries[index].destination_id / world_size;
				int conveyor_index = deliveries[index].current_conveyor_id;

				// Sets the last accessed time of the runway and releases it
				transit_centers[transit_center_index].conveyor_belts[conveyor_index].last_access = current_time;
				atomicExch(&transit_centers[transit_center_index].conveyor_belts[conveyor_index].status, 0);
			}
		}

	} 
}

/* sim_kernelLaunch launches the kernel from the host.
 */
extern "C" bool sim_kernelLaunch(int * outgoing_deliveries_count, unsigned int current_time, int hybrid)
{
	if (hybrid == 1) {
		sim_kernel<<<32,32>>>(deliveries, transit_centers, num_deliveries, num_transit_centers, incoming_deliveries, outgoing_deliveries, outgoing_deliveries_count, world_size, current_time);
	} else {
		sim_kernel<<<1,1>>>(deliveries, transit_centers, num_deliveries, num_transit_centers, incoming_deliveries, outgoing_deliveries, outgoing_deliveries_count, world_size, current_time);
	}

    // Calls this to guarantee that the kernel will finish computation before swapping
    cudaDeviceSynchronize();
    
    return true;    
}
