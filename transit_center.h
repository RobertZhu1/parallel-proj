#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>
#include<string.h>

#define clock_frequency 512000000
#define LINE_SIZE 128

// Transit center that deliveries would travel between
struct transit_center {
	int id; // transit center id
	struct conveyor_belt* conveyor_belts; // array of conveyor belts
	int num_conveyor_belts; // number of conveyor belts in the transit center
};

// A single conveyor_belt in a transit center. Status is used to ensure one delivery processed at a time on conveyor belt.
struct conveyor_belt {
	int id;
	int status;
	int last_access;
};

// Delivery in simulation
struct delivery {
	int id;
	int source_id;
	int destination_id;
	int status;
	int current_conveyor_id;
	unsigned int starting_time;
	unsigned int arriving_time;
	unsigned int transit_time;
	unsigned int processing_time;
	unsigned int wait_time;
};

// Statuses of a delivery in simulation
enum statuses {
	LOST_DELIVERY = -1,
	WAITING = 0,
	STAMPED = 1,
	LEAVE_SOURCE = 2,
	IN_TRANSIT = 3,
	ARRIVE_TO_DES = 4,
	PROCESSING = 5,
	DELIVERED = 6
};
