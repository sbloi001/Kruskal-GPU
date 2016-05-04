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

// GPU function to square root values
__global__ void parallelMergeSort(int size) {
	//int startingPos = threadIdx.x * size * 3;
	//int finalPos = startingPos + size * 3;
   //int element = blockIdx.x*blockDim.x + threadIdx.x;
   //if (element < N) a[element] = sqrt(a[element]);
}


__global__ void update(int *matrix, size_t pitch,int numVert,int * list){
	int reference = (*(list+0));
	int col = blockIdx.y * blockDim.y + (threadIdx.y + 1);	
	int row = blockIdx.x * blockDim.x + (threadIdx.x + 1);
	int* row_ptr;
	if((col <= numVert) && (row <= numVert) ){
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

	int col = blockIdx.x*blockDim.x + threadIdx.x + 1;;
	
	if((col <= numVert) && col != 0){
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
__global__ void getValue(Edge* edges,int pos, int * list){
	Edge temp = (Edge)(*(edges + pos));
	*(list + 0) = temp.orig;
	*(list + 1) = temp.dest;
	//printf("First vertice is %d\n Second vertice is %d\n",*(list + 0),*(list + 1));
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

__global__ void reset(){
	devFound = 0;
}
__global__ void arrayCheck(int * array,int numVert){
	int i;
	
	devFound = 0;
	for(i = 1; i <= numVert;i++){
		if(array[i] == 1){
			devFound =1;
			//printf("Found!\n`");
		}else{
			//printf("Not found!\n");
		}
		array[i] = 0;
	}
	/*
	int pos = blockIdx.x * 	blockDim.x + threadIdx.x;
	
	if( (pos != 0) && (pos <= numVert) && (array[pos] == 1)){
				devFound = 1;
	}
	
	__syncthreads();
	*/
}

__global__ void insertResultingEdge(Edge* original, Edge* destination,int dest_pos, int orig_pos){
	Edge temp = (Edge)(*(original + orig_pos));

	(*(destination + dest_pos)) = temp;
	
	__syncthreads();
}

__global__ void printEdges(Edge* edges,int n){
	int i;
	int min = 0;
	int orig, dest,weight;
	for(i = 0;i < n;i++){
		Edge temp = (Edge)(*(edges + i));
		orig = temp.orig;
		dest = temp.dest;
		weight = temp.weight;
		
		min += weight;
		printf("%d-%d: %d\n",orig,dest, weight);
		
	}
	
	printf("\n\tMinimum cost = %d\n",min);
	__syncthreads();
}

__global__ void adjustGraph(unsigned char* source, unsigned char* dest, int numVert){
	int col = blockIdx.y * blockDim.y + (threadIdx.y + 1);	
	int row = blockIdx.x * blockDim.x + (threadIdx.x + 1);
	
	*(dest + (col - 1) * numVert + (row - 1)) = *(dest + (col * numVert + row));
}

__global__ void copyingGraphs(unsigned char* source, unsigned char* dest, int numVert){
	int col = blockIdx.y * blockDim.y + (threadIdx.y + 1);	
	int row = blockIdx.x * blockDim.x + (threadIdx.x + 1);
	
	*(dest + col * numVert + row) = *(dest + (col * numVert + row));
}


void gpu(unsigned char** graph, int numVert) {	
	int size_1 = (numVert + 1) * (numVert + 1);
	int size = numVert * numVert;
	int* unionMatrix;	
	
	int* checkArray;
	
	size_t pitch;	
	
	unsigned char* d_weights_1;
	unsigned char* d_weights_original;
	unsigned char* d_weights_copy;
	
	int* d_order;
	
	int* vertList; //it is gonna be only size 2.
	
	/****************************************************************************************
	* Alocating memory in the device
	*****************************************************************************************/
	
	cudaMalloc(&d_weights_1, size_1 * sizeof(unsigned char)); //this is  directly copy from the original array that contains numVert + 1 Rows and Cols
	cudaMalloc(&d_weights_original, size * sizeof(unsigned char)); //Array without the extra cols and rows
	cudaMalloc(&d_weights_copy, size * sizeof(unsigned char)); //Array that is gonna be used in the sort
	
	cudaMalloc(&d_order, size * sizeof(int));//would store a sorted array of number to keep track of the indexes to move
	
	cudaMalloc(&vertList,2 * sizeof(int));
	
	cudaMallocPitch(&unionMatrix, &pitch,
                (numVert + 1) * sizeof(Edge), numVert + 1);
	cudaMalloc(&checkArray, (numVert+1)*sizeof(int));

	
	cudaMemcpy(d_weights_1, (*graph), size_1 * sizeof(unsigned char), cudaMemcpyHostToDevice);
	
	/****************************************************************************************
	* End allocating Memory in the device
	*****************************************************************************************/
	int numThreads = 16;
	int numBlocks = numVert / numThreads + 1;
	
	dim3 threadsPerBlock(500,500);
	dim3 numBlocks2D(numVert/threadsPerBlock.x + 1,numVert/threadsPerBlock.y + 1);
	
	adjustGraph<<<numBlocks2D,threadsPerBlock>>>(d_weights_1, d_weights_original, numVert);
	copyingGraphs<<<numBlocks2D,threadsPerBlock>>>(d_weights_original, d_weights_copy, numVert);
	/****************************************************************************************
	* Sorting Section
	*****************************************************************************************/
	
	thrust::device_ptr<unsigned char> t_weight_copy(d_weights_copy);
	thrust::device_ptr<int> t_order(d_order);

	thrust::sort_by_key( t_weights_copy , t_weights_copy + size, t_order);
	
	/****************************************************************************************
	* End Sorting
	*****************************************************************************************/
	
	int j; 
	typeof(devFound) found;
	
	//__device__ int devFound;
	
	//cudaMemcpyToSymbol(devFound,&value,sizeof(int));
	
		
	//printf("Number of Blocks %d\n",numBlocks2D.x);	
	initializeUnionMatrix<<<numBlocks2D,threadsPerBlock>>>(unionMatrix,pitch,numVert);	
	//printGPUMatrix<<<1,1>>>(unionMatrix,numVert + 1);	
	
	//cudaMemcpyToSymbol(devFound, found,sizeof(int));
	int counter = 0;
	
	for(j = 0;j < numEdges;j++){
		getValue<<<1,1>>>(d_edges,j,vertList);

		
		checkSet<<<numBlocks,numThreads>>>(unionMatrix,checkArray,pitch,numVert,vertList);
		
		
		/***************************************************************************************
		* Inserting the node after it was checked that it didnt exist
		****************************************************************************************/
		//reset<<<1,1>>>();
		arrayCheck<<<1,1>>>( checkArray,numVert);
		
		
        cudaMemcpyFromSymbol(&found, devFound, sizeof(found), 0, cudaMemcpyDeviceToHost);
		
		if(found == 0){
			
			insertResultingEdge<<<1,1>>>(d_edges,d_resultEdges,counter,j);
			
			counter++;
			int numThreads_or = threadsPerBlock.x;
			int numBlocks_or = numBlocks2D.x;
		
			setValue<<<1,1>>>(unionMatrix,vertList, pitch);	
		
			//updating the rest of the sets
			orCol<<<numBlocks_or,numThreads_or>>>(unionMatrix,pitch,numVert,vertList);
		
			//printGPUMatrix<<<1,1>>>(unionMatrix,numVert);	

			update<<<numBlocks,threadsPerBlock>>>(unionMatrix,pitch,numVert,vertList);	
			
		}
		
	}
	
	
	printEdges<<<1,1>>>(d_resultEdges,numVert - 1);
	
	cudaMemcpy(h_edges,d_edges, numEdges * sizeof(Edge), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_weights ,d_weights, numEdges * sizeof(int), cudaMemcpyDeviceToHost);
	
	memcpy(((*graph) -> weights) -> weights, h_weights,numEdges * sizeof(int));
	memcpy( ((*graph) -> edges) , h_edges,numEdges * sizeof(Edge));
	 
	cudaFree(d_weights); 
	cudaFree(d_edges);
	cudaFree(d_resultEdges);
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
	unsigned char* theGraph;
	int numVert = 10;
	theGraph = genGraph(numVert,(unsigned) time(&t));

	
	
	
	printf("==============================================\n");
	print2DArray(theGraph,numVert);	
	
	
	printf("==============================================\n");
	printf("==============================================\n");
	printf("Doing Normal\n");
	
	normal(&theGraph,numVert);
	//normal(&a,n);
	//gpu(a,n);
	
	
	//gpu(&theGraph);
	
	//printGraph(theGraph);
	
  return 0;
}

