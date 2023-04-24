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

	// send  recv array init
	cudaMallocManaged(&outgoing_deliveries, (world_size * sizeof(struct delivery *)));
	cudaMallocManaged(&incoming_deliveries, recv_size * sizeof(struct delivery));
	for (int i = 0; i < world_size; i++) cudaMallocManaged(&outgoing_deliveries[i], send_size * sizeof(struct delivery));
	
	// deliveries & transit centers init
	cudaMallocManaged(&deliveries, max_deliveries * sizeof(struct delivery));
	cudaMallocManaged(&transit_centers, max_transit_centers * sizeof(struct transit_center));
	for (int i = 0; i < num_transit_centers; i++) {
		transit_centers[i].num_conveyor_belts = 1;
		cudaMallocManaged(&transit_centers[i].conveyor_belts, transit_centers[i].num_conveyor_belts * sizeof(struct conveyor_belt));
	}
	for (int i = 0; i < num_deliveries; i++) deliveries[i].status = LOST_DELIVERY;

}

__global__ void sim_kernel(struct delivery * deliveries, struct transit_center * transit_centers, unsigned int num_deliveries, 
	unsigned int num_transit_centers, struct delivery * recv_flights, struct delivery ** send_flights, int * count_sent, 
	int world_size, unsigned int current_time){
    // kernel function, simulates one unit of time

	int index;
	for (index = blockIdx.x * blockDim.x + threadIdx.x; index < num_deliveries; index += blockDim.x * gridDim.x) {
		// loop over deliveries, compute their next status
		if (deliveries[index].status == LOST_DELIVERY) {
			// check if can add new deliveries
			int recv_index = atomicSub(&count_sent[world_size], 1) - 1;
			if (recv_index >= 0) memcpy(&deliveries[index], &recv_flights[recv_index], sizeof(struct delivery));
		}

		int delivery_status = deliveries[index].status;

		if (delivery_status == WAITING && deliveries[index].starting_time <= current_time) {
			// reach starting time, delivery status change from waiting to stamped
			deliveries[index].status = STAMPED;
			delivery_status = STAMPED;
		}

		if (delivery_status == STAMPED) {
			int transit_center_index = deliveries[index].source_id / world_size;

			for (int i = 0; i < transit_centers[transit_center_index].num_conveyor_belts; i++) {
				// try to get a conveyer belt
				if (!atomicCAS(&transit_centers[transit_center_index].conveyor_belts[i].status, 0, 1)) {
					if (transit_centers[transit_center_index].conveyor_belts[i].last_access != current_time) {
						// found available conveyer belt, delivery status change from stamped to leave source
						deliveries[index].status = LEAVE_SOURCE;
						deliveries[index].current_conveyor_id = i;
						deliveries[index].processing_time = 1;
						deliveries[index].processing_time--;
						break;
					}
					else atomicExch(&transit_centers[transit_center_index].conveyor_belts[i].status, 0);
				}
			}

			// failed to get conveyer belt, wait time += 1
			if (deliveries[index].status == STAMPED) deliveries[index].wait_time++;
		}

		else if (delivery_status == LEAVE_SOURCE) {
			// when delivery status = leave source, switch to next status in transit when finish processing
			deliveries[index].processing_time--;

			if (deliveries[index].processing_time == 0) {
				// finished processing, status switch to in transit
				deliveries[index].status = IN_TRANSIT;
				int transit_center_index = deliveries[index].source_id / world_size;
				int conveyor_index = deliveries[index].current_conveyor_id;

				// conveyer belt status update to idle, last access update to current time
				transit_centers[transit_center_index].conveyor_belts[conveyor_index].last_access = current_time;
				atomicExch(&transit_centers[transit_center_index].conveyor_belts[conveyor_index].status, 0);
			}
		}

		else if (delivery_status == IN_TRANSIT) {
			// delivery status in transit
			deliveries[index].transit_time--;
			if (deliveries[index].transit_time == 0) {
				// finish transiting delivery, status switch to arrive to des
				deliveries[index].status = ARRIVE_TO_DES;

				// get destination rank and send message from current rank to destination
				int current_rank = deliveries[index].source_id % world_size;
				int dest_rank = deliveries[index].destination_id % world_size;

				int send_index = atomicAdd(&count_sent[dest_rank], 1);
				if (current_rank != dest_rank) {
					memcpy(&send_flights[dest_rank][send_index], &deliveries[index], sizeof(struct delivery));
					deliveries[index].status = LOST_DELIVERY;
				}
			}
		}

		else if (delivery_status == ARRIVE_TO_DES) {
			// delivery status arrive to des
			int transit_center_index = deliveries[index].destination_id / world_size;
			for (int i = 0; i < transit_centers[transit_center_index].num_conveyor_belts; i++) {
				// request a conveyer belt after arriving destination transit center
				if (!atomicCAS(&transit_centers[transit_center_index].conveyor_belts[i].status, 0, 1)) {
					if (transit_centers[transit_center_index].conveyor_belts[i].last_access != current_time) {
						// found idle conveyer belt
						deliveries[index].status = PROCESSING;
						deliveries[index].current_conveyor_id = i;
						deliveries[index].processing_time = 1;
						deliveries[index].processing_time--;
						break;
					}
					else atomicExch(&transit_centers[transit_center_index].conveyor_belts[i].status, 0);
				}
			}

			// failed to find idle conveyer belt, wait time += 1
			if (deliveries[index].status == ARRIVE_TO_DES) deliveries[index].wait_time++;
		}

		else if (delivery_status == PROCESSING) {
			// after arriving destination, delivery needs process before delivered
			deliveries[index].processing_time--;
			if (deliveries[index].processing_time == 0) {
				// finish processing delivery, status switch to delivered
				deliveries[index].status = DELIVERED;
				deliveries[index].arriving_time = current_time + 1;
				int transit_center_index = deliveries[index].destination_id / world_size;
				int conveyor_index = deliveries[index].current_conveyor_id;

				// conveyer belt status update to idle, last access update to current time
				transit_centers[transit_center_index].conveyor_belts[conveyor_index].last_access = current_time;
				atomicExch(&transit_centers[transit_center_index].conveyor_belts[conveyor_index].status, 0);
			}
		}

	} 
}

extern "C" bool sim_kernelLaunch(int * outgoing_deliveries_count, unsigned int current_time, int hybrid)
{
	if (hybrid == 1) {
		// MPI & cuda, parallel code
		sim_kernel<<<32,32>>>(deliveries, transit_centers, num_deliveries, num_transit_centers, incoming_deliveries, outgoing_deliveries, outgoing_deliveries_count, world_size, current_time);
	} 
	else {
		// MPI only, serial code
		sim_kernel<<<1,1>>>(deliveries, transit_centers, num_deliveries, num_transit_centers, incoming_deliveries, outgoing_deliveries, outgoing_deliveries_count, world_size, current_time);
	}

    cudaDeviceSynchronize();
    
    return true;    
}
