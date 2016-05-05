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

#define MAX_WEIGHT  100
#define MAX_VERTICES  20



typedef struct weights{
	int weights[]; 
}Weights;
typedef struct edge{
	int orig;
	int dest;
	int weight;
}Edge;

typedef struct graph{
	int numVert;
	int numEdges;
	Weights* weights;
	Edge edges[] ; 

	
	
} Graph;

struct timeval start, end;

void starttime() {
  gettimeofday( &start, 0 );
}

void endtime(const char* c) {
   gettimeofday( &end, 0 );
   double elapsed = ( end.tv_sec - start.tv_sec ) * 1000.0 + ( end.tv_usec - start.tv_usec ) / 1000.0;
   printf("%s: %f ms\n", c, elapsed); 
}


__global__ void update(int *matrix, size_t pitch,int numVert,int * list){
	int reference = (*(list+0));
	int col = blockIdx.y * blockDim.y + (threadIdx.y );	
	int row = blockIdx.x * blockDim.x + (threadIdx.x );
	int* row_ptr;
	if((col < numVert) && (row <= numVert) ){
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
}

/*
	This methods or 2 columns in the unionMatrix array. This 
	garantees that both sets have the same values.
*/
__global__ void orCol(int *matrix,size_t pitch,int numVert,int * list){
	int firstVert = (*(list + 0));
	int secondVert = (*(list + 1));
	
	int row = blockIdx.x * blockDim.x + (threadIdx.x + 1);
	if(row <= numVert){
		
		int* row_ptr = (int*)((char*)matrix + row * pitch);
		
		int firstValue = row_ptr[firstVert];
		int secondValue = row_ptr[secondVert];
		
		if(firstValue == 1)
			row_ptr[secondVert] = 1;
		if(secondValue == 1)
			row_ptr[firstVert] = 1;		
		}
		
		 __syncthreads();
}
__global__ void checkSet(int* matrix,int* checkArray, size_t pitch, int numVert,int * list){

	int col = blockIdx.x*blockDim.x + threadIdx.x;
	
	if((col < numVert)){
		int* rowFirstVert = (int*)((char*)matrix + (*(list+0)) * pitch);
		int firstValue = rowFirstVert[col];
		
		int* rowSecondVert = (int*)((char*)matrix + (*(list+1)) * pitch);
		int secondValue = rowSecondVert[col];
		
		if((firstValue == 1) && (secondValue == 1)){
			checkArray[col] = 1;
		}else{
			//printf("Not found from thread:%d!\n",threadIdx.x);
		}
			
	}
	
	__syncthreads();
	
}

/*
void insert (int *matrix, size_t pitch,int numVert,int * list){
	
}

*/
__global__ void getValue(unsigned char* original,int* ordered, int * list,int pos, int numVert){
	int index = *(ordered + pos);
	int row = index / numVert;
	int col = index - (row * numVert);
	
	//this is how we retrieve the data back in the CPU
	*(list + 0) = row;
	*(list + 1) = col;

}

__global__ void setValue(int* matrix,int* list,size_t pitch){
			//inserting values in the matrix
		int* ptr = (int*)((char*)matrix + (*(list+0)) * pitch);
		ptr[(*(list+1))] = 1;
		
		ptr = (int*)((char*)matrix +(*(list+1)) * pitch);
		ptr[(*(list+0))] = 1;
}

__global__ void printGPUMatrix(int * matrix,int numVert){
	int i;
	
	for(i = 1; i <= (numVert+1) * (numVert +1); i++){
		printf("%d ",matrix[i]);
	}
}

__global__ void initializeUnionMatrix(int * matrix,size_t pitch, int numVert){
	int col = blockIdx.y * blockDim.y + (threadIdx.y + 1);	
	int row = blockIdx.x * blockDim.x + (threadIdx.x + 1);
	
	
	
	if(row <= numVert && col <= numVert){
		
		if(row == col){
			int* row_ptr= (int*)((char*)matrix + row * pitch);
			//printf("working on col:%d row:%d VALUE:%d\n",col,row,1);
			row_ptr[col] = 1;
		}else{
			//printf("working on col:%d row:%d VALUE:%d\n",col,row,0);
			int* row_ptr= (int*)((char*)matrix + row * pitch);
			row_ptr[col] = 0;
		}
	}
	
	__syncthreads();
}


__device__ int devFound;
__device__ int dev_totalCost;


__global__ void arrayCheck(int * array,int numVert){
	
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	int result = *(array + pos);
	if(result == 1){
		devFound = 0;	
	}	
	
	__syncthreads();
}

/*
* Copy and array of char to another array of char with and offset of 1 to the left.
*/
__global__ void adjustGraph(unsigned char* source, unsigned char* dest, int numVert){
	int col = blockIdx.y * blockDim.y + (threadIdx.y + 1);	
	int row = blockIdx.x * blockDim.x + (threadIdx.x + 1);
	
	if((col < numVert + 1) && (row < numVert + 1))
		*(dest + (col - 1) * numVert + (row - 1)) = *(dest + (col * numVert + row));
	
	__syncthreads();
}

__global__ void copyingGraphs(unsigned char* source, int* dest, int numVert){
	int col = blockIdx.y * blockDim.y + (threadIdx.y );	
	int row = blockIdx.x * blockDim.x + (threadIdx.x );
	
	//Avoiding going out of the array
	if(col < numVert && row < numVert)
		*(dest + col * numVert + row) = (int)*(dest + (col * numVert + row));
	
	__syncthreads();
}

__global__ void addToMinWeight(unsigned char* original,int* list, int numVert){
	int row = *(list + 0);
	int col = *(list + 1);
	
	int weight = *(original + row * numVert + col);
	
	dev_totalCost += weight;
}
/*
* 	Fill out the array of in an ordered way
*/
__global__ void fillOrder(int* order, int size){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if(index < size)
		*(order + index) = index;
	
	__syncthreads();
}

__global__ void printChars(unsigned char* array, int size,int numVert){
	int i;
	
	for(i = 0; i < size; i++){
		
		//if((i % (numVert)) == 0)
			//printf("\n");
		printf("%3d ",*(array + i));
	}
	printf("\n");
	
}
void gpu(unsigned char* graph, int numVert) {	
	int size_1 = (numVert + 1) * (numVert + 1);
	int size = numVert * numVert;
	int* unionMatrix;	
	
	int* checkArray;
	
	size_t pitch;	
	
	unsigned char* d_weights_1;
	unsigned char* d_weights_original;
	int* d_weights_copy;
	
	int* d_order;
	
	int* vertList; //it is gonna be only size 2.
	
	/****************************************************************************************
	* Alocating memory in the device
	*****************************************************************************************/
	
	cudaMalloc(&d_weights_1, size_1 * sizeof(unsigned char)); //this is  directly copy from the original array that contains numVert + 1 Rows and Cols
	cudaMalloc(&d_weights_original, size * sizeof(unsigned char)); //Array without the extra cols and rows
	cudaMalloc(&d_weights_copy, size * sizeof(int)); //Array that is gonna be used in the sort
	
	cudaMalloc(&d_order, size * sizeof(int));//would store a sorted array of number to keep track of the indexes to move
	
	cudaMalloc(&vertList,2 * sizeof(int));
	
	cudaMallocPitch(&unionMatrix, &pitch,
                (numVert) * sizeof(int), numVert);
	cudaMalloc(&checkArray, (numVert)*sizeof(int));

	
	cudaMemcpy(d_weights_1, graph, size_1 * sizeof(unsigned char), cudaMemcpyHostToDevice);
	
	/****************************************************************************************
	* End allocating Memory in the device
	*****************************************************************************************/
	int numThreads = 16;
	int numBlocks = numVert / numThreads + 1;
	
	dim3 threadsPerBlock(500,500);
	dim3 numBlocks2D(numVert/threadsPerBlock.x + 1,numVert/threadsPerBlock.y + 1);
	
	fillOrder<<<numBlocks2D,threadsPerBlock>>>(d_order,size);
	adjustGraph<<<numBlocks2D,threadsPerBlock>>>(d_weights_1, d_weights_original, numVert);
	copyingGraphs<<<numBlocks2D,threadsPerBlock>>>(d_weights_original, d_weights_copy, numVert);
	
	/****************************************************************************************
	* Sorting Section
	*****************************************************************************************/
	
	printChars<<<1,1>>>(d_weights_1,size_1,numVert + 1);
	thrust::device_ptr<int> t_weights_copy(d_weights_copy);
	thrust::device_ptr<int> t_order(d_order);
	

	//thrust::sort_by_key(t_weights_copy , t_weights_copy + size, t_order);
	
	/****************************************************************************************
	* End Sorting
	*****************************************************************************************/
	typeof(devFound) found;
	int totalCost;
	
	initializeUnionMatrix<<<numBlocks2D,threadsPerBlock>>>(unionMatrix,pitch,numVert); //initialize UnionMatrix to all 0s	
	
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
			int numThreads_or = threadsPerBlock.x;
			int numBlocks_or = numBlocks2D.x;
		
			//updating unionMatrix
			setValue<<<1,1>>>(unionMatrix,vertList, pitch);	
		
			//Or both inserted vertices's columns
			orCol<<<numBlocks_or,numThreads_or>>>(unionMatrix,pitch,numVert,vertList);
		
			//Freaki fast union find
			update<<<numBlocks,threadsPerBlock>>>(unionMatrix,pitch,numVert,vertList);
			
		}
		
	
	}
	

	cudaMemcpyFromSymbol(&totalCost, dev_totalCost, sizeof(found), 0, cudaMemcpyDeviceToHost);
	
	printf("\n\tMinimum cost = %d\n",totalCost);
	//cudaMemcpy((*graph),d_result, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	
	cudaFree(vertList); 	
	cudaFree(d_weights_1); 
	cudaFree(d_weights_copy);
	//cudaFree(d_result);
	cudaFree(d_order);
	cudaFree(checkArray);
	cudaFree(unionMatrix);
}

 /*
 int getWeight(int array[],int row, int col, int n){
	 return array[row*n + col];
 }
 
 void setWeight(int* array[],int row, int col, int n, int value){
	 (*array)[row*n + col] = value;
 }
 */

void normal(unsigned char** cost, int n)
{
	int parent[n + 1];
	
	int i,j,a,b,u,v,ne=1;
	int mincost=0;
	unsigned char min;
	//initializing the parent array to all be -1
	for(i = 0; i <= n; i++){
		parent[i] = 0;
	}
	
	while(ne < n)
	{
		for(i=1,min=MAX_WEIGHT;i<=n;i++)
		{
			for(j=1;j <= n;j++)
			{
				if( (*( (*cost) + i * n + j ))< min)
				{
					min= (*( (*cost) + i * n + j ));
					a=u=i;
					b=v=j;
				}
			}
		}
		u=find(parent,u);
		v=find(parent,v);
		if(uni(parent,u,v))
		{
			//ne++;
			printf("%d edge (%d,%d) =%d\n",ne++,a,b,min);
			mincost += (int)min;
		}
		 (*( (*cost) + a * n + b ))=(*( (*cost) + b * n + a ))=MAX_WEIGHT + 1;
	}
	printf("\n\tMinimum cost = %d\n",mincost);
}
int find(int* parent,int i)
{
	while(parent[i])
	i=parent[i];
	return i;
}

int uni(int* parent, int i,int j)
{
	if(i!=j)
	{
		parent[j]=i;
		return 1;
	}
	return 0;
}


/*
	This generates a graph randomly. The array goes in the format ||V1|V2|weight||V2|V3|weight||...||
*/
unsigned char* genGraph(int numVert,unsigned int seed){
	int total_size = (numVert + 1) * (numVert + 1);
	unsigned char* graph = (unsigned char*)malloc(total_size * sizeof(char));
	
	srand(seed);
	int i,j;
	
	for(i = 1; i <= numVert - 1; i++){
		for(j = i + 1; j<= numVert;j++){
			unsigned char temp = (unsigned char)((rand() % MAX_WEIGHT) + 1);
			(*(graph + i * numVert + j)) = temp;
			(*(graph + j * numVert + i)) = temp;
		}
	}
	
	//diagonal more than the maximun weight
	for(i = 1; i <= numVert; i++){
		(*(graph + i * numVert + i)) = MAX_WEIGHT + 1;
	}
	
	return graph;
	
}

/*
	Printing the nodes of the graph as a test
*/
void printGraph(Graph* graph){
	int numEdges = graph -> numEdges;
	int i;

	for(i =0; i < numEdges ;i++){
		printf("%d-%d: %d weight: %d\n",(graph -> edges)[i].orig,(graph -> edges)[i].dest, (graph -> edges)[i].weight, ((graph -> weights) -> weights)[i]);
	}
}


void print2DArray(unsigned char* array, int numVert){
	int i,j;
	
	for(i = 1; i <= numVert; i++){
		for(j = 1; j <= numVert; j++){
			printf("%d-%d: %d\n",i,j,*(array + i*numVert +j));
		}
	}
}

int main()                                                                                                                                                                                  
{         
	time_t t;
	time(&t);
	unsigned char* theGraph;
	int numVert = 5;
	theGraph = genGraph(numVert,(unsigned) t);

	
	int i;
	
	for(i = 0; i < 36; i++){
		printf("%3d ",*(theGraph + i));
	}
	printf("\n");
	
	printf("==============================================\n");
	//print2DArray(theGraph,numVert);	
	
	
	printf("==============================================\n");
	printf("==============================================\n");
	printf("Doing Normal\n");
	
	starttime();
	normal(&theGraph,numVert);
	
	endtime("CPU Time");
	
	
	//printf("THE TIIIIIMEEEE: %d\n", (int)t);
	//normal(&a,n);
	//gpu(a,n);
	
	
	gpu(theGraph, numVert);
	
	//printGraph(theGraph);
	
  return 0;
}

