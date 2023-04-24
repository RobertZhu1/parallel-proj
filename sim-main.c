#include<mpi.h>
#include "transit_center.h"
#include "clockcycle.h"

// The transit centers accessible at each rank
struct transit_center* transit_centers;
int num_transit_centers;

// The deliveries managed current MPI rank
struct delivery *deliveries;
int num_deliveries;

// The deliveries going to be received from or sent to other MPI ranks
struct delivery **outgoing_deliveries;
struct delivery *incoming_deliveries;

int world_rank = 0; // Rank number of current MPI rank
int world_size = 0; // Total number of MPI ranks

// Functions defined in the cuda file (externed with C)
extern void sim_initMaster();
extern bool sim_kernelLaunch(int* outgoing_deliveries_count, unsigned int current_time, int hybrid);

/***********************************************************************************************************/
// Input: None
// Output: MPI_Datatype delivery_struct
// Purpose: Translate the delivery_struct into a MPI datatype
/***********************************************************************************************************/
MPI_Datatype getDeliveryDatatype() {
    int block_lengths[10] = {1,1,1,1,1,1,1,1,1,1};
    MPI_Datatype types[10] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_UNSIGNED, MPI_UNSIGNED, MPI_UNSIGNED, MPI_UNSIGNED, MPI_UNSIGNED};
    MPI_Aint relloc[10];
    relloc[0] = offsetof(struct delivery, id);
    relloc[1] = offsetof(struct delivery, source_id);
    relloc[2] = offsetof(struct delivery, destination_id);
    relloc[3] = offsetof(struct delivery, status);
    relloc[4] = offsetof(struct delivery, current_conveyor_id);
    relloc[5] = offsetof(struct delivery, starting_time);
    relloc[6] = offsetof(struct delivery, arriving_time);
    relloc[7] = offsetof(struct delivery, transit_time);
    relloc[8] = offsetof(struct delivery, processing_time);
    relloc[9] = offsetof(struct delivery, wait_time);

    MPI_Datatype delivery_struct;
    MPI_Type_create_struct(10, block_lengths, relloc, types, &delivery_struct);
    MPI_Type_commit(&delivery_struct);

    return delivery_struct;
}

/***********************************************************************************************************/
// Input: char pointer filename
// Output: None
// Purpose: Read in deliveries at the the assigned slot of input file using parallel MPI I/O
/***********************************************************************************************************/
void MPI_read_input(char* file_name) {
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
			if(field_number == 5) { // line process finished, add data to delivery struct
				deliveries[i].id = id;
                deliveries[i].source_id = source / world_size * world_size + world_rank; // make sure the source location belongs to the rank
                deliveries[i].destination_id = destination;
                deliveries[i].status = WAITING;
                deliveries[i].current_conveyor_id = -1;
                deliveries[i].starting_time = depart_time;
                deliveries[i].transit_time = arrival_time - depart_time;
                deliveries[i].processing_time = 0;
                deliveries[i].wait_time = 0;
			}
		}
	}
	
	if(world_rank == 0) {
		printf("MPI I/O Input Time: %f seconds\n", (clock_now() - start_time) / clock_frequency);
	}
	// ---------- Timed Portion Ends ----------
}

/***********************************************************************************************************/
// Input: integer num_deliveries
// Output: None
// Purpose: Write the processed deliveries to the assigned slot of output file using parallel MPI I/O
/***********************************************************************************************************/
void MPI_write_output(int num_deliveries) {
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
		struct delivery delivery = deliveries[i];
		if(delivery.status == -1) { // invalid delivery
			continue;
		} else { // save the delievery data to rank buffer
			rank_buffer[ind] = delivery.id;
			rank_buffer[ind+1] = delivery.source_id;
			rank_buffer[ind+2] = delivery.destination_id;
			rank_buffer[ind+3] = delivery.status;
			rank_buffer[ind+4] = delivery.starting_time;
			rank_buffer[ind+5] = delivery.arriving_time;
			rank_buffer[ind+6] = delivery.transit_time;
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

/***********************************************************************************************************/
// Input: integer hybrid
// Output: None
// Purpose: Simulate status of each delivery at discrete time step using MPI Communication and COUDA
/***********************************************************************************************************/
void MPI_simulate_deliveries(int hybrid) {
	// Initialization of MPI communication for deliveries across ranks
	MPI_Datatype delivery_struct = getDeliveryDatatype();
	MPI_Request request[2*(world_size-1)];
    MPI_Status status[2*(world_size-1)];
    int incoming_deliveries_count[world_size];
    int outgoing_deliveries_count[world_size+1];    
    for(int i = 0; i < world_size; i++){
        outgoing_deliveries_count[i] = 0;
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
                MPI_Irecv(&incoming_deliveries_count[i], 1, MPI_INT, i, t, MPI_COMM_WORLD, &request[j]);
				// Notify the other ranks of the number of deliveries they should receive from rank i
                MPI_Isend(&outgoing_deliveries_count[i], 1, MPI_INT, i, t, MPI_COMM_WORLD, &request[j+world_size-1]);
                j++;
            }
        }
        MPI_Waitall(2*world_size-2, request, status); // Blocks and waits for all updates has completed

        int deliveries_received = 0;
        for (i = 0, j = 0; i < world_size; i++) {
            if(i != world_rank){
				// Rank i gets the deliveries from the other ranks according the the delivery count updates received previously
                MPI_Irecv(&incoming_deliveries[deliveries_received], incoming_deliveries_count[i], delivery_struct, i, t, MPI_COMM_WORLD, &request[j]);
				// Rank i sends the deliveries to the other ranks according the the delivery count updates it made previously
                MPI_Isend(outgoing_deliveries[i], outgoing_deliveries_count[i], delivery_struct, i, t, MPI_COMM_WORLD, &request[j+world_size-1]);
                deliveries_received += incoming_deliveries_count[i];
                j++;
            }
        }
        MPI_Waitall(2*world_size-2, request, status); // Blocks and waits for all deliveries to be exchanged

        // Resets delivery update counts to 0 at the beginning of each discrete step simulation
        for(i = 0; i < world_size; i++){
            outgoing_deliveries_count[i] = 0;
        }
        outgoing_deliveries_count[world_size] = deliveries_received; // Store the total number of deliveries moved between ranks

        // Launches the kernel of delivery status update in hybrid or nonhybrid mode
		sim_kernelLaunch(outgoing_deliveries_count, t, hybrid);
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
    //int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	//numranks = world_size;
    
    // Get the rank of the process
    //int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	//myrank = world_rank;

    if (argc < 4 && world_rank == 0) { // check command line arguments
        printf("Simulation requires at least three arguments: filename, number of deliveries, and number of tansit centers.\n");
        return 1;
    }

    // Reads command line arguments
	char* filename = NULL;
    filename = argv[1];
    num_deliveries = atoi(argv[2]);
    num_transit_centers = atoi(argv[3]);

	// Initializes CUDA resources at each rank
    sim_initMaster();
	// Read in the deliveries from input file
	MPI_read_input(filename);
	// Simulate the deliveries
	if (argc == 5 && strcmp(argv[4], "h") == 0) { // simulate with hybrid mode
		MPI_simulate_deliveries(1);
	} else { // simulate with MPI only
		MPI_simulate_deliveries(0);
	}
	// Write the delivery status at end of simulation to output file
	MPI_write_output(num_deliveries);

	MPI_Finalize();
}
