#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>
#include<mpi.h>
#include<string.h>
#include "airport.h"
#include "clockcycle.h"

#define clock_frequency 512000000
#define LINE_SIZE 128

// The airports of each rank are stored here
struct airport *airports;
int num_airports;

// The flights departing from airports in current MPI rank
struct flight *flights;
int num_flights;

// The flights being sent to and rcvd from are stored here
struct flight **g_send_flights;
struct flight *g_recv_flights;

int myrank = 0; // Rank number of current MPI rank
int numranks = 0; // Total number of MPI ranks

// Functions defined in the cuda file (externed with C)
extern void sim_initMaster();
extern bool sim_kernelLaunch(int* count_to_send, unsigned int current_time, int hybrid);

MPI_Datatype getDeliveryDatatype(){
    int block_lengths[10] = {1,1,1,1,1,1,1,1,1,1};
    MPI_Datatype types[10] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_UNSIGNED, MPI_UNSIGNED, MPI_UNSIGNED, MPI_UNSIGNED, MPI_UNSIGNED};
    MPI_Aint relloc[10];
    relloc[0] = offsetof(struct flight, id);
    relloc[1] = offsetof(struct flight, source_id);
    relloc[2] = offsetof(struct flight, destination_id);
    relloc[3] = offsetof(struct flight, stage);
    relloc[4] = offsetof(struct flight, current_runway_id);
    relloc[5] = offsetof(struct flight, starting_time);
    relloc[6] = offsetof(struct flight, landing_time);
    relloc[7] = offsetof(struct flight, travel_time);
    relloc[8] = offsetof(struct flight, taxi_time);
    relloc[9] = offsetof(struct flight, wait_time);

    MPI_Datatype delivery_struct;
    MPI_Type_create_struct(10, block_lengths, relloc, types, &delivery_struct);
    MPI_Type_commit(&delivery_struct);

    return delivery_struct;
}

void MPI_read_input(char* file_name, int world_rank, int world_size) {
	double start_time;
	// ---------- Timed Portion Starts ----------
	if (world_rank == 0) {
		printf("Parallel discrete event simulation for scheduled deliveries.\n");
		start_time = clock_now();
	}

	// Open the input file with MPI
	MPI_File input_file;
	MPI_Status file_status;
	MPI_Offset file_size;
	MPI_File_open(MPI_COMM_WORLD, file_name, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
	MPI_File_get_size(input_file, &file_size);

	// File evenly split among the ranks, each rank read in a portion of the file
	int rank_buffer_size = file_size / world_size;
	int num_chars;
	char* rank_buffer;
	rank_buffer = (char*) calloc(rank_buffer_size+1, sizeof(char));
	MPI_File_set_view(input_file, world_rank*rank_buffer_size*sizeof(char), MPI_CHAR, MPI_CHAR, "native", MPI_INFO_NULL);	
	MPI_File_read(input_file, rank_buffer, rank_buffer_size, MPI_CHAR, &file_status);
	MPI_Get_count(&file_status, MPI_CHAR, &num_chars);
	MPI_File_close(&input_file);
	rank_buffer[num_chars] = '\0';

	// Process the input by line and store in to delivery struct
	char line_buffer[LINE_SIZE];
	for(int i = 0; i < rank_buffer_size/LINE_SIZE; i++) {
		for(int j = 0; j < LINE_SIZE; j++) { // load each line in buffer for processing
			line_buffer[j] = rank_buffer[(i*LINE_SIZE)+j];
		}
		int field_number = 0; // number of fields processed in line so far
		int id, source, destination, depart_time, arrival_time = -1;
        char *field = strtok(line_buffer, ","); // fields are separated by ','
		while (field) {
			int number = atoi(field);
			if(field_number == 0) id = number; //id
			if(field_number == 1) source = number; // source of delivery
			if(field_number == 2) destination = number; // destination of delivery
			if(field_number == 3) depart_time = number; // departure time of delievery
			if(field_number == 4) arrival_time = number; // arrival time of delievery
		   	field = strtok(NULL, ",");
			field_number++;
			if(field_number == 5) { // line process finished, add data to flight struct
				flights[i].id = id;
                flights[i].source_id = source / world_size * world_size + world_rank; // make sure the source location belongs to the rank
                flights[i].destination_id = destination;
                flights[i].stage = WAITING;
                flights[i].current_runway_id = -1;
                flights[i].starting_time = depart_time;
                flights[i].travel_time = arrival_time - depart_time;
                flights[i].taxi_time = 0;
                flights[i].wait_time = 0;
			}
		}
	}
	
	if(world_rank == 0) {
		printf("MPI I/O Input Time: %f seconds\n", (clock_now() - start_time) / clock_frequency);
	}
	// ---------- Timed Portion Ends ----------
}

void MPI_write_output(int num_deliveries, int world_rank, int world_size) {
	double start_time;
	// ---------- Timed Portion Starts ----------
	if (world_rank == 0) {
		start_time = clock_now();
	}

	// Open the output file with MPI
	MPI_File output_file;	
	MPI_Status file_status;
	MPI_File_open(MPI_COMM_WORLD, "output", MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    
	// Each rank process its portion of delieveries
	int* rank_buffer = malloc(8*num_deliveries*sizeof(int));
	unsigned int ind = 0;
	for(int i = 0; i < num_deliveries; i++) {
		struct flight delivery = flights[i];
		if(delivery.stage == -1) { // invalid delivery
			continue;
		} else { // save the delievery data to rank buffer
			rank_buffer[ind] = delivery.id;
			rank_buffer[ind+1] = delivery.source_id;
			rank_buffer[ind+2] = delivery.destination_id;
			rank_buffer[ind+3] = delivery.stage;
			rank_buffer[ind+4] = delivery.starting_time;
			rank_buffer[ind+5] = delivery.landing_time;
			rank_buffer[ind+6] = delivery.travel_time;
			rank_buffer[ind+7] = delivery.wait_time;
			ind += 8;
		}
	}

	// Each rank writes its result in the assigned slot of the output file
	MPI_File_seek(output_file, 0, MPI_SEEK_END);
	MPI_File_set_view(output_file, world_rank*8*num_deliveries*sizeof(int), MPI_INT, MPI_INT, "native", MPI_INFO_NULL);
	MPI_File_write(output_file, rank_buffer, ind, MPI_INT, &file_status);
	MPI_File_close(&output_file);

	if (world_rank == 0) {
		printf("MPI I/O Write Time: %f seconds\n", (clock_now() - start_time) / clock_frequency);
	}
	// ---------- Timed Portion Ends ----------
}

void MPI_simulate_deliveries(int hybrid, int world_rank, int world_size) {
	// Initialization of MPI communication for deliveries across ranks
	MPI_Datatype delivery_struct = getDeliveryDatatype();
	MPI_Request request[2*(world_size-1)];
    MPI_Status status[2*(world_size-1)];
    int incoming_deliveries[world_size];
    int outgoing_deliveries[world_size+1];    
    for(int i = 0; i < world_size; i++){
        outgoing_deliveries[i] = 0;
    }

	double start_time;
	// ---------- Timed Portion Starts ----------
	if (world_rank == 0) {
		start_time = clock_now();
	}

    // Simulate the 100 discrete time steps
    int t, i, j = 0;
    for (t = 0; t < 100; t++) {
        for (i = 0, j = 0; i < world_size; i++) {
            if (i != world_rank) {
				// Get the number of deliveries rank i should prepare to receive from the other ranks
                MPI_Irecv(&incoming_deliveries[i], 1, MPI_INT, i, t, MPI_COMM_WORLD, &request[j]);
				// Notify the other ranks of the number of deliveries they should receive from rank i
                MPI_Isend(&outgoing_deliveries[i], 1, MPI_INT, i, t, MPI_COMM_WORLD, &request[j+world_size-1]);
                j++;
            }
        }
        MPI_Waitall(2*world_size-2, request, status); // Blocks and waits for all updates has completed

        int deliveries_received = 0;
        for (i = 0, j = 0; i < world_size; i++) {
            if(i != world_rank){
				// Rank i gets the deliveries from the other ranks according the the delivery count updates received previously
                MPI_Irecv(&g_recv_flights[deliveries_received], incoming_deliveries[i], delivery_struct, i, t, MPI_COMM_WORLD, &request[j]);
				// Rank i sends the deliveries to the other ranks according the the delivery count updates it made previously
                MPI_Isend(g_send_flights[i], outgoing_deliveries[i], delivery_struct, i, t, MPI_COMM_WORLD, &request[j+world_size-1]);
                deliveries_received += incoming_deliveries[i];
                j++;
            }
        }
        MPI_Waitall(2*world_size-2, request, status); // Blocks and waits for all deliveries to be exchanged

        // Resets delivery update counts to 0 at the beginning of each discrete step simulation
        for(i = 0; i < world_size; i++){
            outgoing_deliveries[i] = 0;
        }
        outgoing_deliveries[world_size] = deliveries_received; // Store the total number of deliveries moved between ranks

        // Launches the kernel of delivery status update in hybrid or nonhybrid mode
		sim_kernelLaunch(outgoing_deliveries, t, hybrid);
    }

    MPI_Barrier(MPI_COMM_WORLD); // Synchronize all the ranks before writing to output file
    MPI_Type_free(&delivery_struct);

    if(world_rank == 0) {
		printf("Parallel Descrete Event Simulation Time: %f seconds\n", (clock_now() - start_time) / clock_frequency);
	}
	// ---------- Timed Portion Ends ----------
}

int main(int argc, char *argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	numranks = world_size;
    
    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	myrank = world_rank;

    char* filename = NULL;

    if(argc < 4 && world_rank == 0){
        printf("Simulation requires at least three arguments: filename, number of deliveries, and number of locations.\n");
        return 1;
    }

    // Reads command line arguments
    filename = argv[1];
    num_flights = atoi(argv[2]);
    num_airports = atoi(argv[3]);

	// Initializes the world with the specified pattern
    sim_initMaster();

	MPI_read_input(filename, world_rank, world_size);

	if (argc == 5 && strcmp(argv[4], "h") == 0) {
		MPI_simulate_deliveries(1, world_rank, world_size);
	} else {
		MPI_simulate_deliveries(0, world_rank, world_size);
	}

	MPI_write_output(num_flights, world_rank, world_size);

	MPI_Finalize();
}
