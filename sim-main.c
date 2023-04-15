#define LINE_SIZE 128

#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>
#include<mpi.h>
#include<string.h>
#include<stddef.h>
#include "airport.h"

// The airports of each rank are stored here
struct airport *airports;
int num_airports;

// The flights departing from airports in current MPI rank
struct flight *flights;
int num_flights;

// The flights being sent to and rcvd from are stored here
struct flight **g_send_flights;
struct flight *g_recv_flights; 

// Rank number of current MPI rank
int myrank = 0;

// Total number of MPI ranks
int numranks = 0;

// EXTERNED FUNCTIONS: These are defined in the cuda file
extern void sim_initMaster();

extern bool sim_kernelLaunch(int * count_to_send, unsigned int current_time);

extern void printFlight(struct flight f);

MPI_Datatype getFlightDatatype();

int main(int argc, char *argv[])
{
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &numranks);

    char * filename = NULL;
    unsigned int printOn = 0;

    if(argc != 4 && myrank == 0){
        printf("Simulation requires three arguments: filename, number of flights, and number of airports.\n");
        return 1;
    }

    // Reads command line arguments
    filename = argv[1];
    num_flights = atoi(argv[2]);
    num_airports = atoi(argv[3]);

    double startTime;

	// Sets the initial time after MPI has been initialized
	if(myrank == 0)
	{
		printf("This is a simulation of scheduled flights running in parallel.\n");
		startTime = MPI_Wtime();
	}

	// Initializes the world with the specified pattern
    sim_initMaster();

    MPI_Datatype flight_struct = getFlightDatatype();

    MPI_Request request[2*numranks-2];
    MPI_Status status[2*numranks-2];
    int send_count[numranks];
    int count_to_send[numranks+1];
    
    for(int i = 0; i < numranks; i++){
        count_to_send[i] = 0;
    }
    // initialize file	
	MPI_File file, outfile;	
	MPI_Status file_status, file_status2;
	MPI_Offset filesize;	

	MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
	MPI_File_get_size(file, &filesize);

	MPI_File_open(MPI_COMM_WORLD, "output.txt", MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &outfile);

	int bufsize = filesize/numranks;
	int nchars;
	
	char* cfile;
	char line[LINE_SIZE];

	cfile = (char*) malloc((bufsize+1)*sizeof(char));
	MPI_File_set_view(file, myrank*bufsize*sizeof(char), MPI_CHAR, MPI_CHAR, "native", MPI_INFO_NULL);	
	MPI_File_read(file, cfile, bufsize, MPI_CHAR, &file_status);
	MPI_Get_count(&file_status, MPI_CHAR, &nchars);
	cfile[nchars] = (char)0;
	//printf("%s\n", cfile);
	//printf("Rank %d:\n", myrank);
	for(int i = 0; i < bufsize/LINE_SIZE; i++)
	{
		for(int j = 0; j < LINE_SIZE; j++)
		{
			line[j] = cfile[(i*LINE_SIZE)+j];
		}

		int field_n = 0;
		int id, src_airport, dst_airport, depart_time, arrival_time = -1;
        	char *field = strtok(line, ",");
		while (field) 
		{
			int num = atoi(field);
			
			if(field_n == 0)	// id
			{
			//	printf("id:%d\n", num);
				id = num;
			}
			if(field_n == 1)	// source airport
			{
			//	printf("src:%d\n", num);
				src_airport = num;
			}
			if(field_n == 2)	// destination airport
			{
			//	printf("dst:%d\n", num);
				dst_airport = num;
			}
			if(field_n == 3)	// depart time
			{
			//	printf("dep:%d\n", num);
				depart_time = num;
			}
			if(field_n == 4)	// arrival time
			{
			//	printf("arr:%d\n", num);
				arrival_time = num;	
			}
		   	field = strtok(NULL, ",");
			field_n++;
			if(field_n == 5)
			{
				// set up struct and send to array here
				flights[i].id = id;
                flights[i].source_id = src_airport / numranks * numranks + myrank;
                flights[i].destination_id = dst_airport;
                flights[i].stage = WAITING;
                flights[i].current_runway_id = -1;
                flights[i].starting_time = depart_time;
                flights[i].travel_time = arrival_time - depart_time;
                flights[i].taxi_time = 0;
                flights[i].wait_time = 0;
			}
		}
		//printf("%d,%d,%d,%d,%d\n", id, src_airport, dst_airport, depart_time, arrival_time);
	}
	
    // Iterates through iterations and computes flights
    int t, i, j;
    for(t = 0; t < 100; t++)
    {
        // Sends and receives the number of flights that will be sent
        for(i = 0, j = 0; i < numranks; i++){
            if(i != myrank){
                MPI_Irecv(&send_count[i], 1, MPI_INT, i, t, MPI_COMM_WORLD, &request[j]);
                MPI_Isend(&count_to_send[i], 1, MPI_INT, i, t, MPI_COMM_WORLD, &request[j+numranks-1]);
                j++;
            }
        }

        MPI_Waitall(2*numranks-2, request, status);

        // Sends and receives the flights from every other rank
        int offset = 0;
        for(i = 0, j = 0; i < numranks; i++){
            if(i != myrank){
                MPI_Irecv(&g_recv_flights[offset], send_count[i], flight_struct, i, t, MPI_COMM_WORLD, &request[j]);
                MPI_Isend(g_send_flights[i], count_to_send[i], flight_struct, i, t, MPI_COMM_WORLD, &request[j+numranks-1]);
                offset += send_count[i];
                j++;
            }
        }

        MPI_Waitall(2*numranks-2, request, status);

        // Resets count_to_send array to 0
        for(i = 0; i < numranks; i++){
            count_to_send[i] = 0;
        }

        // Last index is used as a counter to indicate the total number of flights rcvd
        count_to_send[numranks] = offset;

        // Launches the kernel
    	sim_kernelLaunch(count_to_send, t);

    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Prints the total time elapsed for the program
    if(myrank == 0)
    {
    	printf("Time elapsed: %f seconds\n", MPI_Wtime() - startTime);
    }
	
	//Write code used for testing write I/O speed in comments below
	//MPI_File_seek(outfile, 0, MPI_SEEK_END);
	//MPI_File_set_view(outfile, myrank * num_flights * sizeof(struct flight), flight_struct, flight_struct, "native", MPI_INFO_NULL);
    //MPI_File_write(outfile, flights, num_flights, flight_struct, MPI_STATUS_IGNORE);
    
	double outputStartTime;
	if(myrank == 0)
		outputStartTime = MPI_Wtime();
	for(int i = 0; i < num_flights; i++)
	{
		
		struct flight f = flights[i];
		if(f.stage == -1)
			continue;
		char buffer[1024];	
		int n = sprintf(buffer, "%d %d %d %d %u %u %u %u\n", f.id, f.source_id, f.destination_id, f.stage, 
			f.starting_time, f.landing_time, f.travel_time, f.wait_time);
		buffer[n] = '\0';
		//MPI_File_set_view(file, myrank*(num_flights/numranks)*sizeof(char), MPI_CHAR, MPI_CHAR, "native", MPI_INFO_NULL);	
		MPI_File_seek(outfile, 0, MPI_SEEK_END);
		MPI_File_write(outfile, buffer, n, MPI_CHAR, &file_status2);
	}
	
	if(myrank == 0)
		printf("Outputting to File Time Elapsed: %f seconds\n", MPI_Wtime() - outputStartTime);
	
	MPI_File_close(&file);
	MPI_File_close(&outfile);
	
    MPI_Type_free(&flight_struct);
	MPI_Finalize();
}


MPI_Datatype getFlightDatatype(){
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

    MPI_Datatype flight_struct;
    MPI_Type_create_struct(10, block_lengths, relloc, types, &flight_struct);
    MPI_Type_commit(&flight_struct);

    return flight_struct;
}
