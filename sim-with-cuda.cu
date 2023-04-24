
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include "airport.h"

extern struct transit_center *transit_centers;
extern int num_transit_centers;

extern struct delivery *deliveries;
extern int num_deliveries;

extern struct delivery **outgoing_deliveries;
extern struct delivery *incoming_deliveries; 

extern int world_rank; // current rank

extern int world_size; // total rank

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
	int delivery_lim = 6000000;
	int serv_loc_lim = 361440;

	// send and recv init
	cudaMallocManaged(&outgoing_deliveries, (world_size*sizeof(struct delivery*)));
	cudaMallocManaged(&incoming_deliveries, recv_size*sizeof(struct delivery));
	for (int i = 0; i < world_size; i++) cudaMallocManaged(&outgoing_deliveries[i], send_size*sizeof(struct delivery));

	// deliveries and service location init
	cudaMallocManaged(&deliveries, delivery_lim*sizeof(struct delivery));
	cudaMallocManaged(&transit_centers, serv_loc_lim*sizeof(struct transit_center));
	for (int i = 0; i < num_transit_centers; i++) {
		transit_centers[i].num_conveyor_belts = 1;
		cudaMallocManaged(&transit_centers[i].conveyor_belt, transit_centers[i].num_conveyor_belts * sizeof(struct conveyor_belt));
	}
	for (int i = 0; i < num_deliveries; i++) deliveries[i].status = LOST_DELIVERY;

}

__global__ void sim_kernel(struct delivery *deli, struct transit_center *serv_loc, unsigned int deli_lim, 
	struct delivery *recv_deli, struct delivery **send_deli, int *count_sent, int world_size, 
	unsigned int current_time){
   	// kernel function, simulates 1 time unit

	int ind;
	for (ind = blockIdx.x * blockDim.x + threadIdx.x; ind < deli_lim; ind += blockDim.x * gridDim.x) {
		// loop through all deliveries, compute next status
		if (deli[ind].status == LOST_DELIVERY) {
			// Checks recv deliveries to see if any new deliveries can be added in
			int recv_index = atomicSub(&count_sent[world_size], 1) - 1;
			if (recv_index >= 0) memcpy(&deli[ind], &recv_deli[recv_index], sizeof(struct delivery));
		}

		int deli_stage = deli[ind].status;
		if (deli_stage == WAITING && deli[ind].starting_time <= current_time) {
			// current time = takeoff time, set to next status
			deli[ind].status = STAMPED;
			deli_stage = STAMPED;
		}

		int serv_loc_ind = deli[ind].source_id / world_size;
		if (deli_stage == STAMPED) {
			// if preparing to start delivery, wait until resource alloc
			
			for (int i = 0; i < serv_loc[serv_loc_ind].num_conveyor_belts; i++) {
				// check if can start delivery now
				if (!atomicCAS(&serv_loc[serv_loc_ind].conveyor_belt[i].status, 0, 1)) {
					if (serv_loc[serv_loc_ind].conveyor_belt[i].last_access != current_time) {
						deli[ind].status = LEAVE_SOURCE;
						deli[ind].current_conveyor_id = i;
						deli[ind].processing_time = 1;
						deli[ind].processing_time--;
						break;
					}
					else atomicExch(&serv_loc[serv_loc_ind].conveyor_belt[i].status, 0);
				}
			}

			if (deli[ind].status == STAMPED) deli[ind].wait_time++;
			// cannot start delivery now, wait for next time unit
		}
		else if (deli_stage == LEAVE_SOURCE){
			deli[ind].processing_time--;

			// Advances the status to cruising if the delivery has completed taxiing
			if (deli[ind].processing_time == 0) {
				deli[ind].status = IN_TRANSIT;
				int conveyor_belt_ind = deli[ind].current_conveyor_id;

				// Sets last access time of conveyor_belt and releases conveyor_belt
				serv_loc[serv_loc_ind].conveyor_belt[conveyor_belt_ind].last_access = current_time;
				atomicExch(&serv_loc[serv_loc_ind].conveyor_belt[conveyor_belt_ind].status, 0);
			}
		}
		else if (deli_stage == IN_TRANSIT) {
			deli[ind].transit_time--;
			if (deli[ind].transit_time == 0) {
				deli[ind].status = DELIVERED;

				// Gets the appropriate rank to send to
				int current_rank = deli[ind].source_id % world_size;
				int dest_rank = deli[ind].destination_id % world_size;

				// Gets the index to write into and increments the number of elements
				int send_index = atomicAdd(&count_sent[dest_rank], 1);
				if (current_rank != dest_rank) {
					memcpy(&send_deli[dest_rank][send_index], &deli[ind], sizeof(struct delivery));
					deli[ind].status = LOST_DELIVERY;
				}
			}
		}
		else if (deli_stage == DELIVERED) {
			// Attempts to obtain a conveyor_belt in order to land
			for (int i = 0; i < serv_loc[serv_loc_ind].num_conveyor_belts; i++) {
				if (!atomicCAS(&serv_loc[serv_loc_ind].conveyor_belt[i].status, 0, 1)) {
					if(serv_loc[serv_loc_ind].conveyor_belt[i].last_access != current_time){
						deli[ind].status = PROCESSING;
						deli[ind].current_conveyor_id = i;
						deli[ind].processing_time = 1;
						deli[ind].processing_time--;
						break;
					}
					else atomicExch(&serv_loc[serv_loc_ind].conveyor_belt[i].status, 0);
				}
			}

			// Increments wait time if delivery could not get conveyor_belt
			if (deli[ind].status == DELIVERED) {
				deli[ind].wait_time++;
			}
		}
		else if (deli_stage == PROCESSING) {
			deli[ind].processing_time--;
			if (deli[ind].processing_time == 0) {
				deli[ind].status = LANDED;
				deli[ind].arriving_time = current_time + 1;
				int conveyor_belt_ind = deli[ind].current_conveyor_id;

				// Sets the last accessed time of the conveyor_belt and releases it
				serv_loc[serv_loc_ind].conveyor_belt[conveyor_belt_ind].last_access = current_time;
				atomicExch(&serv_loc[serv_loc_ind].conveyor_belt[conveyor_belt_ind].status, 0);
			}
		}

	} 
}

/* sim_kernelLaunch launches the kernel from the host.
 */
extern "C" bool sim_kernelLaunch(int * count_to_send, unsigned int current_time, int hybrid)
{
	if (hybrid == 1) sim_kernel<<<32,32>>>(deliveries, transit_centers, num_deliveries, incoming_deliveries, outgoing_deliveries, count_to_send, world_size, current_time);
	else sim_kernel<<<1,1>>>(deliveries, transit_centers, num_deliveries, incoming_deliveries, outgoing_deliveries, count_to_send, world_size, current_time);

    // Calls this to guarantee that the kernel will finish computation before swapping
    cudaDeviceSynchronize();
    
    return true;    
}

