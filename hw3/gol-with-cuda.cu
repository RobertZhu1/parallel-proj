
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>
#include<cuda.h>
#include<cuda_runtime.h>

// Result from last compute of world.
extern unsigned char *g_resultData;

// Current state of world.
extern unsigned char *g_data;

// Ghost row above world section
extern unsigned char *g_upperRow;

// Ghost row below world section
extern unsigned char *g_lowerRow;

// Current width of world.
extern size_t g_worldWidth;

/// Current height of world.
extern size_t g_worldHeight;

/// Current data length (product of width and height)
extern size_t g_dataLength;

// Rank number of current MPI rank
extern int myrank;

// Total number of MPI ranks
extern int numranks;

extern "C" void gol_initAllZeros( size_t worldWidth, size_t worldHeight )
{
    int i;

    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    // Creates shared memory between the host and device
    cudaMallocManaged( &g_data, (g_dataLength * sizeof(unsigned char)));
    cudaMallocManaged( &g_resultData, (g_dataLength * sizeof(unsigned char))); 

    // Initializes all members of g_data to 0
    for( i = 0; i < g_dataLength; i++)
    {
        g_data[i] = 0;
    }
}

extern "C" void gol_initAllOnes( size_t worldWidth, size_t worldHeight )
{
    int i;
    
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    cudaMallocManaged( &g_data, (g_dataLength * sizeof(unsigned char)));

    // set all rows of world to true
    for( i = 0; i < g_dataLength; i++)
    {
	   g_data[i] = 1;
    }
    
    cudaMallocManaged( &g_resultData, (g_dataLength * sizeof(unsigned char))); 
}

extern "C" void gol_initOnesInMiddle( size_t worldWidth, size_t worldHeight )
{
    int i;
    
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    cudaMallocManaged( &g_data, (g_dataLength * sizeof(unsigned char)));

    // Initializes all members of g_data to 0
    for( i = 0; i < g_dataLength; i++)
    {
        g_data[i] = 0;
    }

    // set first 1 rows of world to true
    for( i = (g_worldHeight-1)*g_worldWidth; i < g_worldHeight*g_worldWidth; i++)
    {
	if( (i >= ( (g_worldHeight-1)*g_worldWidth + 128)) && (i < ((g_worldHeight-1)*g_worldWidth + 138)))
	{
	    g_data[i] = 1;
	}
    }
    
    cudaMallocManaged( &g_resultData, (g_dataLength * sizeof(unsigned char))); 
}

extern "C" void gol_initOnesAtCorners( size_t worldWidth, size_t worldHeight )
{
    int i;

    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    cudaMallocManaged( &g_data, (g_dataLength * sizeof(unsigned char)));

    // Initializes all members of g_data to 0
    for( i = 0; i < g_dataLength; i++)
    {
        g_data[i] = 0;
    }

    // Sets corners only for rank 0 and last rank
    if(myrank == 0)
    {
    	g_data[0] = 1; // upper left
    	g_data[worldWidth-1]=1; // upper right
    }
    if(myrank == (numranks - 1)){
    	g_data[(worldHeight * (worldWidth-1))]=1; // lower left
    	g_data[(worldHeight * (worldWidth-1)) + worldWidth-1]=1; // lower right
    }
    
    cudaMallocManaged( &g_resultData, (g_dataLength * sizeof(unsigned char))); 
}

extern "C" void gol_initSpinnerAtCorner( size_t worldWidth, size_t worldHeight )
{
    int i;

    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    cudaMallocManaged( &g_data, (g_dataLength * sizeof(unsigned char)));

    // Initializes all members of g_data to 0
    for( i = 0; i < g_dataLength; i++)
    {
        g_data[i] = 0;
    }

    // Initializes spinner only for rank 0
    if(myrank == 0){
    	g_data[0] = 1; // upper left
    	g_data[1] = 1; // upper left +1
    	g_data[worldWidth-1]=1; // upper right
    }

    cudaMallocManaged( &g_resultData, (g_dataLength * sizeof(unsigned char))); 
}

// The gol_initGhostRows function allocates data for ghost rows and initializes values to 0
extern "C" void gol_initGhostRows( size_t worldWidth )
{
	// Allocates memory for the two ghost rows
	cudaMallocManaged( &g_upperRow, (worldWidth * sizeof( unsigned char)));
	cudaMallocManaged( &g_lowerRow, (worldWidth * sizeof( unsigned char)));

	// Initializes ghost rows to 0
	int i;
	for(i = 0; i < worldWidth; i++)
	{
		g_upperRow[i] = 0;
		g_lowerRow[i] = 0;
	}
}

// The gol_initMaster function has been modified based on the specifications
// in the assignment3 pdf. It has also been modified to call gol_initGhostRows.
extern "C" void gol_initMaster( unsigned int pattern, size_t worldWidth, size_t worldHeight)
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

    // Initializes ghost rows
	gol_initGhostRows( worldWidth );

    switch(pattern)
    {
    case 0:
	gol_initAllZeros( worldWidth, worldHeight );
	break;
	
    case 1:
	gol_initAllOnes( worldWidth, worldHeight );
	break;
	
    case 2:
	gol_initOnesInMiddle( worldWidth, worldHeight );
	break;
	
    case 3:
	gol_initOnesAtCorners( worldWidth, worldHeight );
	break;

    case 4:
	gol_initSpinnerAtCorner( worldWidth, worldHeight );
	break;

    default:
	printf("Pattern %u has not been implemented \n", pattern);
	exit(-1);
    }
}

// gol_swap takes two pointers to unsigned char pointers and swaps them
extern "C" void gol_swap( unsigned char **pA, unsigned char **pB)
{
    // Swaps the pointers using a temporary variable
    unsigned char *tmp = *pA;
    *pA = *pB;
    *pB = tmp;
}

// This function was provided by Dr. Carothers in HW #1. It has been
// modified in order to print to a separate file.
extern "C" void gol_printWorld(FILE *output)
{
    int i, j;

    for( i = 0; i < g_worldHeight; i++)
    {
	fprintf(output, "Row %2d: ", i);
	for( j = 0; j < g_worldWidth; j++)
	{
	    fprintf(output,"%u ", (unsigned int)g_data[(i*g_worldWidth) + j]);
	}
	fprintf(output, "\n");
    }

    fprintf(output, "\n\n");
}

/* gol_kernel is the kernel function called from the host. 
   It simulates one run of the Game of Life given current grid d_data and 
   ghost rows and stores its computation in d_resultData.
 */
__global__ void gol_kernel(const unsigned char* d_data,
						  unsigned int worldWidth,
						  unsigned int worldHeight,
						  unsigned char* d_resultData,
						  const unsigned char *d_upperRow,
						  const unsigned char *d_lowerRow){
	int index;

    // Iterates through the locations in the GOL world
    for(index = blockIdx.x * blockDim.x + threadIdx.x; index < worldWidth*worldHeight; index += blockDim.x * gridDim.x)
    {
        int x0, x1, x2;
        int y, y0, y1, y2;

        // Computes the current x and neighboring y's and accounts for wraparound
        x1 = index % worldWidth;
        x0 = (x1 + worldWidth - 1) % worldWidth;
        x2 = (x1 + 1) % worldWidth;
        
        // Computes the current y and neighboring y's and accounts for wraparound
        y = index / worldWidth;
        y1 = y * worldWidth;
        y0 = ((y + worldHeight - 1) % worldHeight) * worldWidth;
        y2 = ((y + 1) % worldHeight) * worldWidth;

        // Finds the number of alive cells around the current cell
        unsigned int numAliveCells = 0;

        // Counts number of diagonally adjacent neighbors that are alive
        numAliveCells = numAliveCells + ((y != 0 & d_data[x0+y0]) | (y == 0 & d_upperRow[x0]));
        numAliveCells = numAliveCells + ((y != worldWidth-1 & d_data[x0+y2]) | (y == worldWidth-1 & d_lowerRow[x0]));
        numAliveCells = numAliveCells + ((y != 0 & d_data[x2+y0]) | (y == 0 & d_upperRow[x2]));
        numAliveCells = numAliveCells + ((y != worldWidth-1 & d_data[x2+y2]) | (y == worldWidth-1 & d_lowerRow[x2]));

        // Counts number of non-diagonally adjacent neighbors that are alive
        numAliveCells = numAliveCells + d_data[x0+y1];
        numAliveCells = numAliveCells + ((y != 0 & d_data[x1+y0]) | (y == 0 & d_upperRow[x1]));
        numAliveCells = numAliveCells + ((y != worldWidth-1 & d_data[x1+y2]) | (y == worldWidth-1 & d_lowerRow[x1]));
        numAliveCells = numAliveCells + d_data[x2+y1];

        // Sets cell in resultData array to alive (1) or dead (0) based on number of alive cells around it
        if(numAliveCells == 2){
            d_resultData[x1+y1] = d_data[x1+y1];
        }
        else if(numAliveCells == 3){
            d_resultData[x1+y1] = 1;
        }
        else{
            d_resultData[x1+y1] = 0;
        }
    }
}

/* gol_kernelLaunch launches the kernel from the host.
   threadsCount is the number of threads to specify in the kernel call.
 */
extern "C" bool gol_kernelLaunch(unsigned char** d_data, 
                    unsigned char** d_resultData, 
                    size_t worldWidth, 
                    size_t worldHeight, 
                    size_t iterationsCount, 
                    ushort threadsCount)
{
    if(worldWidth * worldHeight / threadsCount == 0)
    {
         gol_kernel<<<1,threadsCount>>>(*d_data, worldWidth, worldHeight, *d_resultData, g_upperRow, g_lowerRow);    
    }
    else
    {
        gol_kernel<<<worldWidth * worldHeight / threadsCount,threadsCount>>>(*d_data, worldWidth, worldHeight, *d_resultData, g_upperRow, g_lowerRow);
    }

    // Calls this to guarantee that the kernel will finish computation before swapping
    cudaDeviceSynchronize();

    // Updates g_data with our newly computed world
    gol_swap(d_data, d_resultData);

    return true;    
}