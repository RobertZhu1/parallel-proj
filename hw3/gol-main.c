
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>
#include<mpi.h>
#include<string.h>

// Result from last compute of world.
unsigned char *g_resultData = NULL;

// Current state of world.
unsigned char *g_data = NULL;

// Row above current world section
unsigned char *g_upperRow = NULL;

// Row below current world section
unsigned char *g_lowerRow = NULL;

// Current width of world.
size_t g_worldWidth = 0;

/// Current height of world.
size_t g_worldHeight = 0;

/// Current data length (product of width and height)
size_t g_dataLength = 0;

// Rank number of current MPI rank
int myrank = 0;

// Total number of MPI ranks
int numranks = 0;

// EXTERNED FUNCTIONS: These are defined in the cuda file
extern void gol_initAllZeros( size_t worldWidth, size_t worldHeight );

extern void gol_initAllOnes( size_t worldWidth, size_t worldHeight );

extern void gol_initOnesInMiddle( size_t worldWidth, size_t worldHeight );

extern void gol_initOnesAtCorners( size_t worldWidth, size_t worldHeight );

extern void gol_initSpinnerAtCorner( size_t worldWidth, size_t worldHeight );

extern void gol_initGhostRows( size_t worldWidth);

extern void gol_initMaster( unsigned int pattern, size_t worldWidth, size_t worldHeight);

extern void gol_swap( unsigned char **pA, unsigned char **pB);

extern void gol_printWorld(FILE *output);

extern bool gol_kernelLaunch(unsigned char** d_data, 
                    unsigned char** d_resultData, 
                    size_t worldWidth, 
                    size_t worldHeight, 
                    size_t iterationsCount, 
                    ushort threadsCount);

int main(int argc, char *argv[])
{
	unsigned int pattern = 0;
    unsigned int worldSize = 0;
    unsigned int itterations = 0;
    unsigned int threadsCount = 0;
    unsigned int printOn = 0;

    double startTime;

    if( argc != 6 )
    {
		printf("GOL requires 5 arguments: pattern number, sq size of the world, number of itterations, thread count, and whether printing is on. e.g. ./gol 0 32 2 64 0\n");
		exit(-1);
    }

    // Reads command line arguments
    pattern = atoi(argv[1]);
    worldSize = atoi(argv[2]);
    itterations = atoi(argv[3]);
    threadsCount = atoi(argv[4]);
    printOn = atoi(argv[5]);

	// Initialize MPI
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &numranks);

	// Sets the initial time after MPI has been initialized
	if(myrank == 0)
	{
		printf("This is the Game of Life running in parallel.\n");
		startTime = MPI_Wtime();
	}

	// Initializes the world with the specified pattern
    gol_initMaster(pattern, worldSize, worldSize);

    MPI_Request request[4];
    MPI_Status status[4];

    // Iterates through iterations and computes world
    int i;
    for(i = 0; i < itterations; i++)
    {
    	// Receives ghost rows from neighboring ranks
    	MPI_Irecv(g_upperRow, g_worldWidth, MPI_CHAR, (myrank + numranks - 1) % numranks,
    		  i, MPI_COMM_WORLD, &request[0]);
    	MPI_Irecv(g_lowerRow, g_worldWidth, MPI_CHAR, (myrank + 1) % numranks,
    		  i + itterations + 1024, MPI_COMM_WORLD, &request[1]);

    	// Sends ghost rows to neighboring ranks
    	MPI_Isend(g_data, g_worldWidth, MPI_CHAR, (myrank + numranks - 1) % numranks,
    		  i + itterations + 1024, MPI_COMM_WORLD, &request[2]);
    	MPI_Isend(g_data + (g_worldHeight-1)*g_worldWidth, g_worldWidth, MPI_CHAR, (myrank + 1) % numranks,
    		  i, MPI_COMM_WORLD, &request[3]);
    	MPI_Waitall(4, request, status);

    	// Computes one iteration of the world
    	gol_kernelLaunch(&g_data, &g_resultData, worldSize, worldSize, itterations, threadsCount);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Prints the total time elapsed for the program
    if(myrank == 0)
    {
    	printf("Time elapsed: %f seconds\n", MPI_Wtime() - startTime);
    }

    // Prints to separate files if print is specified
    if(printOn)
    {
        // Creates array to store file name ("output-xx.txt")
        char file[14];
        strcpy(file, "output-");
        if(myrank < 10){
            file[7] = '0';
            file[8] = '0' + myrank;
        }
        else if(myrank < 100){
            file[7] = '0' + (myrank / 10);
            file[8] = '0' + (myrank % 10);
        }
        file[13] = '\0';
        strcpy(file+9, ".txt");

        // Opens file and prints to it
        FILE *output = fopen(file, "w+");
        fprintf(output, "Rank %d:\n", myrank);
    	gol_printWorld(output);
        fclose(output);
    }

	MPI_Finalize();
}
