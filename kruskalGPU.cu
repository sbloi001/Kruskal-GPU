#include <emmintrin.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

int find(int*,int);
int uni(int*,int,int);

int getWeight(int array[],int row, int col, int n);
int setWeight(int* array[],int row, int col, int n, int value);

#define MAX_WEIGHT  250

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

		
/****************************************************************************************
* Global variables declaration
****************************************************************************************/
__device__ int devFound; //global variable to get result of the check method

__device__ int dev_totalCost; //cumulative minimun weight

struct timeval start, end;


/****************************************************************************************
* Tracking time section
****************************************************************************************/

//starts the timer
void starttime() {
  gettimeofday( &start, 0 );
}

//stops the timer and then print the desire message with the time
void endtime(const char* c) {
   gettimeofday( &end, 0 );
   double elapsed = ( end.tv_sec - start.tv_sec ) * 1000.0 + ( end.tv_usec - start.tv_usec ) / 1000.0;
   printf("%s: %f ms\n", c, elapsed); 
}

/****************************************************************************************
* End of Tracking time section
****************************************************************************************/


/****************************************************************************************
* Union find implementation section
****************************************************************************************/


/*
*	This method initialize a union-find-matrix  with the diagonal set to 1.
*	This matrix keeps track of sets of vertices connected to and specific vertice.
*	Each column means the set of vertices that are connected to the vertices denoted by the column number.
*	
*	Ex. 3 x 3 adjecency matrix initialized where each node belongs to its set of connected vertices. 0 connected to 0
*	1 connected to 1 and so on.
*	
*	indexes | 0 | 1 | 2 |
*	--------------------
*		0   | 1 | 0 | 0 | 
*		----------------
*		1   | 0	| 1 | 0 |
*		----------------
*		2   | 0	| 0 | 1 |
*		
*		
*	Ex. 3 x 3 adjecency matrix after some insertions. Here in column 0 we have row 0 to mark as 1 and row 1 mark as 1. This
*	means that  vertices 0 and 1 are connected to vertice 0 (column number)
*	
*	indexes | 0 | 1 | 2 |
*	--------------------
*		0   | 1 | 1 | 0 | 
*		----------------
*		1   | 1	| 1 | 0 |
*		----------------
*		2   | 0	| 0 | 1 |	
*/
__global__ void initializeUnionMatrix(int * matrix,size_t pitch, int numVert){
	int row = blockIdx.y * blockDim.y + threadIdx.y;	
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	
	//printf("working on col:%d row:%d \n",col,row);
	
	if(row < numVert && col < numVert){
		
		if(row == col){ //if is t he diagonal make mark it as 1. Each element belongs to its set.
			
			int* row_ptr= (int*)((char*)matrix + row * pitch);
			
			row_ptr[col] = 1;
			
			//printf("working on col:%d row:%d VALUE:%d\n",col,row,1);
			
		}else{
			
			//printf("working on col:%d row:%d VALUE:%d\n",col,row,0);
			int* row_ptr= (int*)((char*)matrix + row * pitch);
			row_ptr[col] = 0;
			
		}
	}
	
	
	__syncthreads();
	
}// end initializeUnionMatrix

/*
*	This method makes the first update in the union-find-matrix
*	reflecting the fact that the new inserted vertices  are connected 
*	between them
*
*	Arguments:
*
*	int* matrix: pointer to the union-find-matrix
*	size_t pitch: size (in bytes) of each row
*	int* list: Mainly a size 2 array with the 2 vertice numbers to be updated.
*/
__global__ void setValue(int* matrix,int* list,size_t pitch){
		//inserting values in the union-find-matrix
		int* ptr = (int*)((char*)matrix + (*(list+0)) * pitch);
		ptr[(*(list+1))] = 1;
		
		ptr = (int*)((char*)matrix +(*(list+1)) * pitch);
		ptr[(*(list+0))] = 1;
} //end of setValue


/*
*	This methods or 2 columns in the union-find-matrix. This 
*	garantees that both sets have the same values.
*	Arguments:
*	
*	int* matrix: pointer to the union-find-matrix
*	size_t pitch: size (in bytes) of each row
*	int numVert: number of vertices of the adjecency matrix
*	int* list: Mainly a size 2 array with the 2 vertices number to be checked
*/
__global__ void orCol(int *matrix,size_t pitch,int numVert,int * list){
	int firstVert = (*(list + 0));
	int secondVert = (*(list + 1));
	
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(row < numVert){ //checking not to go out of boundries
		
		int* row_ptr = (int*)((char*)matrix + row * pitch);
		
		int firstValue = row_ptr[firstVert];
		int secondValue = row_ptr[secondVert];
		
		//TODO: Work on a better way to or.
		//not a fancy or :(
		if(firstValue == 1)
			row_ptr[secondVert] = 1;
		if(secondValue == 1)
			row_ptr[firstVert] = 1;		
	}
		
	__syncthreads();
}//end orCol



/*
*	This method updates the whole union-find-matrix forcing all the vertices 
*	connected to the new added vertices to be updated to reflec the new connections created.
*	Arguments:
*	
*	int* matrix: pointer to the union-find-matrix
*	size_t pitch: size (in bytes) of each row
*	int numVert: number of vertices of the adjecency matrix
*	int* list: Mainly a size 1 array with the vertice number to be used as reference for the update
*/
__global__ void update(int *matrix, size_t pitch,int numVert,int * list){
	int reference = (*(list+0));
	
	int row = blockIdx.y * blockDim.y + threadIdx.y;	
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	
	int* row_ptr;
	if((col < numVert) && (row < numVert) ){ // checking not to go out of boundries
		if(col != reference){ //Avoiding writing to the reference column
			row_ptr= (int*)((char*)matrix + row * pitch);
			int value = row_ptr[reference];
			
			if(value == 1){//if the current row exists in the reference column set
				row_ptr= (int*)((char*)matrix + col * pitch);
				int exist = row_ptr[reference];
				
				if(exist == 1){
					row_ptr= (int*)((char*)matrix + row * pitch);
					row_ptr[col] = 1;
				}
			}
		}
	}
	
	 __syncthreads();
}//end of update


/****************************************************************************************
* End of Union find implementation section
****************************************************************************************/

/****************************************************************************************
* Avoiding Loops section
****************************************************************************************/

/*
*	This method checks the union-find-matrix to see if 2 vertices are already connected. 
*	It has n threads, being n the number of vertices, each checking a single column.
*
*	Arguments:
*	
*	int* matrix: pointer to the union-find-matrix
*	int* checkArray: pointer to the array where the result of the checking will be
*	size_t pitch: size (in bytes) of each row
*	int numVert: number of vertices of the adjecency matrix
*	int* list: Mainly a size 2 array with the 2 vertices number to be checked
*/
__global__ void checkSet(int* matrix,int* checkArray, size_t pitch, int numVert,int * list){

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	
	if((col < numVert)){ //checking not to go out of boundries in the union-find-matrix
		int* rowFirstVert = (int*)((char*)matrix + (*(list+0)) * pitch);
		int firstValue = rowFirstVert[col];
		
		int* rowSecondVert = (int*)((char*)matrix + (*(list+1)) * pitch);
		int secondValue = rowSecondVert[col];
		
		if((firstValue == 1) && (secondValue == 1)){//if both elements were found in the same set it means that they are connected already
			checkArray[col] = 1; // marking that correspondent spot of the current thread as 1, meaning, found
		}else{
			//printf("Not found from thread:%d!\n",threadIdx.x);
		}
			
	}
	
	__syncthreads();
	
}//end CheckSet


/*
*	This methods check if any thread from the checkSet method found the two vertices candidates for insertion. 
*	
*	Arguments:
*	
*	int* array: pointer to the array where the result of the checking are
*	int numVert: number of vertices of the adjecency matrix
*/
__global__ void arrayCheck(int * array,int numVert){
	
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(pos < numVert){ //checking not to go out of boundries
		int result = *(array + pos);
		
		if(result == 1){ //if some thread reported 1. It means that they belong to the same set, hence, they are connected.
			devFound = 1;	
		}
		
	}
			
	__syncthreads();
}//end arrayCheck


/*
*	Reseting the value of devFound to 0
*/
__global__ void resetGlobalFound(){
	devFound = 0;	
}//end resetGlobalFound


/*
*	Resets the check Array to all 0s.
*	
*	Arguments:
*
*	int* array: pointer to the array where the result of the checking are
*	int numVert: number of vertices of the adjecency matrix
*/
__global__ void resetArrayCheck(int * array,int numVert){
	
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(pos < numVert){
		*(array + pos) = 0;
	}
	
}//end resetArrayCheck

/****************************************************************************************
* End of Avoiding Loops section
****************************************************************************************/


/****************************************************************************************
* Extra tool methods section
****************************************************************************************/

/*
* 	Fill out the array of in an ordered way.
*	
*	Arguments:
*	
*	int* order: array to be filled
*	int size: max lenght of the array
*	int numVert: number of vertices of the adjecency matrix
*/
__global__ void fillOrder(int* order, int size, int numVert){

	int index = blockIdx.x * blockDim.x + threadIdx.x; //getting the actual position in the array.
	
	if(index < size)//checking not to go out of boundries
		*(order + index) = index;
	
	__syncthreads();
	
}//end of fillOrder


/*
* This method creates a copy of the array.
*
* 	Arguments:
*
*	unsigned char* source: Array to be copied
*	unsigned char* dest: Array to copy to
*	int numVert: number of vertices of the adjecency matrix
*/
__global__ void copyingGraphs(unsigned char* source, int* dest, int size){

	int index = blockIdx.x * blockDim.x + threadIdx.x; //getting the actual position in the array.
	
	if(index < size) //checking not to go out of boundries
		*(dest + index) = (int)*(source + index);
	
	__syncthreads();
	
}//end copyingGraphs


/*
*	This method increments the total minimun weight of the new minimun spamming tree
*
*	Arguments:
*
*	unsigned char* original: Original array with the orignial positions for the weights
*	int* list: Mainly a size 2 array with the 2 vertice numbers to get the weight of the edge between them.
*	int numVert: number of vertices of the adjecency matrix
*/
__global__ void addToMinWeight(unsigned char* original,int* list, int numVert){
	int row = *(list + 0);
	int col = *(list + 1);
	
	int weight = *(original + row * numVert + col);
	
	dev_totalCost += weight;
}//end addToMinWeight


/*
*	This method retrieves the column and row of 
*/
__global__ void getValue(unsigned char* original,int* ordered, int * list,int pos, int numVert){
	int index = *(ordered + pos);
	int row = index / numVert;
	int col = index % numVert;
	
	//this is how we retrieve the data back in the CPU
	*(list + 0) = row;
	*(list + 1) = col;

}//end getValue


/*
*	Prints an array in the GPU
*/
__global__ void printA(int* array, int size){
	int i;
	
	for(i = 0; i < size; i++){
		printf("%d ",array[i]);
		
	}
	
	printf("\n");
	
}
/****************************************************************************************
* End of the Extra tool methods section
****************************************************************************************/


/*
*	GPU implementation of Kruskal's algorithm multi-threaded.
*/
void gpu(unsigned char** graph, int numVert) {	
	int size = numVert * numVert;
	int* unionMatrix;	
	
	int* checkArray;
	
	size_t pitch;	
	
	unsigned char* d_weights_original;
	int* d_weights_copy;
	
	int* d_order;
	
	int* vertList; //it is gonna be only size 2.
	
	/****************************************************************************************
	* Alocating memory in the device
	*****************************************************************************************/
	
	cudaMalloc(&d_weights_original, size * sizeof(unsigned char)); //Array without the extra cols and rows
	cudaMalloc(&d_weights_copy, size * sizeof(int)); //Array that is gonna be used in the sort
	
	cudaMalloc(&d_order, size * sizeof(int));//would store a sorted array of number to keep track of the indexes to move
	
	cudaMalloc(&vertList,2 * sizeof(int));
	
	cudaMallocPitch(&unionMatrix, &pitch,
                (numVert) * sizeof(int), numVert); //allocating memory for the union-find-matrix
				
	cudaMalloc(&checkArray, (numVert)*sizeof(int)); //allocating memory for the checkArray

	
	cudaMemcpy(d_weights_original, (*graph), size * sizeof(unsigned char), cudaMemcpyHostToDevice); //Transfering the 1D array from the CPU's DRAM into the Device's DRAM
	
	cudaCheckErrors("cudaMalloc fail");
	/****************************************************************************************
	* End allocating Memory in the device
	*****************************************************************************************/
	int numThreads = 1024;
	int numBlocks = numVert / numThreads + 1;
	
	dim3 threadsPerBlock(30,30);
	dim3 numBlocks2D(numVert/threadsPerBlock.x + 1,numVert/threadsPerBlock.y + 1);
	
	fillOrder<<<size/numThreads + 1,numThreads>>>(d_order,size,numVert);
	cudaCheckErrors("filling arrays fail");
	
	copyingGraphs<<<size/numThreads + 1,numThreads>>>(d_weights_original, d_weights_copy, size);
	cudaCheckErrors("Copying arrays fail");
	
	/****************************************************************************************
	* Sorting Section
	*****************************************************************************************/

	thrust::sort_by_key(thrust::device_ptr<int>(d_weights_copy) , thrust::device_ptr<int>(d_weights_copy + size), thrust::device_ptr<int> (d_order));
	cudaCheckErrors("Sort fail");
	
	/****************************************************************************************
	* End Sorting
	*****************************************************************************************/
	typeof(devFound) found;
	int totalCost;
	
	resetArrayCheck<<<numBlocks,numThreads>>>(checkArray,numVert); //resetting the checking array to all 0s
	
	resetGlobalFound<<<1,1>>>(); //resseting the global found variable to 0
	cudaCheckErrors("Reset Found fail");
	
	initializeUnionMatrix<<<numBlocks2D,threadsPerBlock>>>(unionMatrix,pitch,numVert); //initializing union-find-matrix
	cudaCheckErrors("Union find initialization fail");

	int j; 
	int counter = 0;
	
	for(j = 0;j < size;j++){
		getValue<<<1,1>>>(d_weights_original, d_order, vertList,j,numVert);
		
		//checking if those vertices are not in any set
		checkSet<<<numBlocks,numThreads>>>(unionMatrix,checkArray,pitch,numVert,vertList);
		
	/***************************************************************************************
	* Inserting the node after it was checked that it didnt exist
	****************************************************************************************/
	
		arrayCheck<<<numBlocks,numThreads>>>( checkArray,numVert);
		
		cudaMemcpyFromSymbol(&found, devFound, sizeof(found), 0, cudaMemcpyDeviceToHost);
		
		if(found == 0){
				
			//insertResultingEdge<<<numBlocks,numThreads>>>(d_edges,d_resultEdges,counter,j);
			addToMinWeight<<<1,1>>>(d_weights_original,vertList,numVert);
			
			counter++;

			//updating unionMatrix
			setValue<<<1,1>>>(unionMatrix,vertList, pitch);	
		
			//Or both inserted vertices's columns
			orCol<<<numBlocks,numThreads>>>(unionMatrix,pitch,numVert,vertList);
		
			//Freaki fast union find
			update<<<numBlocks2D,threadsPerBlock>>>(unionMatrix,pitch,numVert,vertList);
			
		}
		
		resetArrayCheck<<<numBlocks,numThreads>>>(checkArray,numVert); //resetting the checking array to all 0s
		resetGlobalFound<<<1,1>>>(); //resseting the global found variable to 0
		
		if(counter == numVert - 1){ //if we got the min spaming tree
			break;
		}
	
	}
	

	cudaMemcpyFromSymbol(&totalCost, dev_totalCost, sizeof(totalCost), 0, cudaMemcpyDeviceToHost);
	
	printf("\n\tMinimum cost = %d\n",totalCost);
	//cudaMemcpy((*graph),d_result, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	
	cudaFree(vertList); 	
	cudaFree(d_weights_copy);
	//cudaFree(d_result);
	cudaFree(d_order);
	cudaFree(checkArray);
	cudaFree(unionMatrix);
}


 /*
	Normal implementation of Kruskal's algorithm single threaded.
 */
void normal(unsigned char* cost, int n)
{
	//array used for the union find algorithm
	int parent[n];
	
	int i,j,a,b,u,v,ne=1;
	int mincost=0;
	unsigned char min;
	
	//initializing the parent array to all be -1
	for(i = 0; i < n; i++){
		parent[i] = -1;
	}
	
	while(ne < n) //stops when it has N - 1 edges inserted
	{
		for(i=0,min=MAX_WEIGHT;i<n;i++) //iterates over the hole matrix finding the minimun weight.
		{
			for(j=0;j < n;j++)
			{
				if( (*(cost + i * n + j ))< min)
				{
					min = (*(cost + i * n + j ));
					a = u = i;
					b = v = j;
				}
			}
		}
		
		u=find(parent,u);
		v=find(parent,v);
		
		if(uni(parent,u,v) != -1)
		{
			ne++;
			//printf("%d edge (%d,%d) =%d\n",ne++,a,b,min);
			mincost += (int)min;
		}
		 (*(cost + a * n + b )) = (*(cost + b * n + a )) = MAX_WEIGHT + 1; //replacing the current weight of the edges for more than the maximun
																		   //so it wouldnt be counted twice
	}
	printf("\n\tMinimum cost = %d\n",mincost);
} //end of normal

/*
	This method find the las vertice to which the imput vertice is connected to.
*/
int find(int* parent,int i)
{
	while(parent[i] != -1){
		
		i = parent[i];		
	}
		
	
	return i;
}//end of find 


/*
	Implementation of the union find for this way of doing Kruskal
*/
int uni(int* parent, int i,int j)
{
	if(i!=j)
	{
		parent[j]=i;
		return 1;
	}
	
	return -1;
}//end of uni


/*
	This generates the adjecency matrix of a full  connected weighted graph. 
	The result is a 1D array. Position 0 of the array being the start of the row 0 column 0.
	The generated matrix doesnt accept loops.
*/
unsigned char* genGraph(int numVert,unsigned int seed){
	
	//total amount of cells of the matrix that translate into the size of the array in 1D
	int total_size = numVert * numVert;
	
	//allocating memory for the array. This is being allocated in the heap.
	unsigned char* graph = (unsigned char*)malloc(total_size * sizeof(char));
	
	//initializating the seed
	srand(seed);
	
	int i,j;
	
	unsigned char temp;
	
	//filling the matrix except the diagonal.
	for(i = 0; i < numVert - 1; i++){
		for(j = i + 1; j <  numVert;j++){
			temp = (unsigned char)((rand() % MAX_WEIGHT) + 1); //generated weight
			(*(graph + i * numVert + j)) = temp; //position (i,j)
			(*(graph + j * numVert + i)) = temp; //position (j,i)
		}
	}
	
	//diagonal more than the maximun weight
	for(i = 0; i < numVert; i++){
		(*(graph + i * numVert + i)) = MAX_WEIGHT + 1; //the diagonal weights more than any other node. It would count.
													   //No loops allowed
	}
	
	return graph; //pointer to the graph in the heap.
	
}//end of genGraph


/*
	It is the driver.
*/
int main()                                                                                                                                                                                  
{         
	time_t t;
	time(&t); //generating a random seed every second
	unsigned char* theGraph; //pointer to a 1D array where the matrix NxN is gonna be
	int numVert = 1000; // number of vertices for the test case
	
	theGraph = genGraph(numVert,(unsigned) t); //initializing the matrix
	
	
	printf("==============================================\n");
	printf("==============================================\n");
	printf("Doing GPU\n");
	
	starttime();
	gpu(&theGraph, numVert);
	
	endtime("GPU Time");
	
	printf("==============================================\n");
	printf("==============================================\n");
	printf("Doing Normal\n");
	
	starttime();
	normal((unsigned char*)theGraph,numVert);
	
	endtime("CPU Time");
	

  return 0;
}

