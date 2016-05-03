#include <emmintrin.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>

int find(int);
int uni(int,int);

int getWeight(int array[],int row, int col, int n);
int setWeight(int* array[],int row, int col, int n, int value);

#define MAX_WEIGHT  100
#define MAX_VERTICES  20

int parent[MAX_VERTICES];

typedef struct edge{
	int orig;
	int dest;
	int weight;
}Edge;

typedef struct graph{
	int numVert;
	int numEdges;
	Edge edges[];
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


__global__ void update(int *matrix, size_t pitch,int numVert,int reference){
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
__global__ void orCol(int *matrix,size_t pitch,int numVert,int firstVert,int secondVert){
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
__global__ void checkSet(int* matrix,int* checkArray, size_t pitch, int numVert,int firstVert, int secondVert){

	int col = blockIdx.x*blockDim.x + threadIdx.x + 1;;
	
	if((col <= numVert) && col != 0){
		int* rowFirstVert = (int*)((char*)matrix + firstVert * pitch);
		int firstValue = rowFirstVert[col];
		
		int* rowSecondVert = (int*)((char*)matrix + secondVert * pitch);
		int secondValue = rowSecondVert[col];
		
		if((firstValue == 1) && (secondValue == 1)){
			checkArray[col] = 1;
		}	
			
	}
	
}

void insert (int *matrix, size_t pitch,int numVert,int firstVert,int secondVert){
	//update part
	dim3 threadsPerBlock(1024,1024);
	dim3 numBlocks(numVert/threadsPerBlock.x + 1,numVert/threadsPerBlock.y + 1);
	
	//checking part
	int numThreads_or = threadsPerBlock.x;
	int numBlocks_or = numBlocks.x;
	
	//inserting values in the matrix
	int* ptr = (int*)((char*)matrix + firstVert * pitch);
	ptr[secondVert] = 1;
	
	ptr = (int*)((char*)matrix +secondVert * pitch);
	ptr[firstVert] = 1;
	
	//updating the rest of the sets
	orCol<<<numBlocks_or,numThreads_or>>>(matrix,pitch,numVert,1,2);
	
	update<<<numBlocks,threadsPerBlock>>>(matrix,pitch,numVert,1);	
}

void gpu(Graph ** graph) {
	int numVert = (*graph) -> numVert;
	int* unionMatrix;
	
	int* checkArray;
	size_t pitch;
	
	cudaMallocPitch(&unionMatrix, &pitch,
                (numVert + 1) * sizeof(Edge), numVert + 1);
	cudaMalloc(&checkArray, (numVert+1)*sizeof(int));
	
	int numThreads = 1024;
	int numBlocks = numVert / numThreads + 1;
	
	int i, found = 0;
	
	for(;;){
		checkSet<<<numBlocks,numThreads>>>(unionMatrix,checkArray,pitch,numVert,1,2);
	
		for(i = 1; i <= numVert; i++){
			if(checkArray[i] == 1){
				found = 1;
			}
			checkArray[i] = 0;
		}
		
		if(found == 0){
			insert(unionMatrix, pitch, numVert, 1,2);
		}
		
		
	}
	
	
	
	
	
	
	
	
	//char allocatedVertices[n];
   //int numThreads = 1024;
   //int numBlocks = N / 1024 + 1;

   //float* gpuA;
   //int* gpuArray;
   //cudaMalloc(allocatedVertices, n*sizeof(char));
   //cudaMalloc(gpuArray, n*n*sizeof(int));
   //cudaMemcpy(gpuArray, array, n*n*sizeof(int), cudaMemcpyHostToDevice);
   //gpu_sqrt<<<numBlocks, numThreads>>>(gpuA, N);
   //cudaMemcpy(a, gpuA, N*sizeof(float), cudaMemcpyDeviceToHost);
   //cudaFree(&gpuA);
}

 /*
 int getWeight(int array[],int row, int col, int n){
	 return array[row*n + col];
 }
 
 void setWeight(int* array[],int row, int col, int n, int value){
	 (*array)[row*n + col] = value;
 }
 */

void normal(int cost[MAX_VERTICES][MAX_VERTICES], int n)
{
	int i,j,a,b,u,v,ne=1;
	int min,mincost=0;
	
	while(ne < n)
	{
		for(i=1,min=999;i<=n;i++)
		{
			for(j=1;j <= n;j++)
			{
				if(cost[i][j] < min)
				{
					min=cost[i][j];
					a=u=i;
					b=v=j;
				}
			}
		}
		u=find(u);
		v=find(v);
		if(uni(u,v))
		{
			printf("%d edge (%d,%d) =%d\n",ne++,a,b,min);
			mincost +=min;
		}
		cost[a][b]=cost[b][a]=999;
	}
	printf("\n\tMinimum cost = %d\n",mincost);
}
int find(int i)
{
	while(parent[i])
	i=parent[i];
	return i;
}

int uni(int i,int j)
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
Graph* genGraph(int numVert,unsigned int seed){
	int numEdges = ((numVert * (numVert - 1))/2);
	Graph *graph;
	graph = (Graph*)malloc(sizeof(Graph) +  numEdges*sizeof(Edge));
	
	graph -> numEdges = numEdges;
	graph -> numVert = numVert;
	
	int i,j,edgeNumber = 0;
	
	//generating seed
	srand(seed);
	
	for(i = 1; i <= numVert - 1; i++){
		for(j = i + 1; j<= numVert;j++){
			Edge* edge = (Edge*)malloc(sizeof(Edge));
			edge -> orig = i;
			edge -> dest = j;
			edge -> weight = (rand() % MAX_WEIGHT) + 1;
			(graph -> edges)[edgeNumber] = (*edge);
			edgeNumber++;
		}
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
		printf("%d-%d: %d\n",(graph -> edges)[i].orig,(graph -> edges)[i].dest, (graph -> edges)[i].weight);
	}
}

void print2DArray(int array[][MAX_VERTICES], int numVert){
	int i,j;
	
	for(i = 1; i <= numVert; i++){
		for(j = 1; j <= numVert; j++){
			printf("%d-%d: %d\n",i,j,(array)[i][j]);
		}
	}
}

int main()                                                                                                                                                                                  
{         
	time_t t;
	Graph* theGraph;
	theGraph = genGraph(3,(unsigned) time(&t));

	
	int numVert = theGraph -> numVert;
	int numEdges = theGraph -> numEdges;
	int theArray[MAX_VERTICES][MAX_VERTICES];
	
	int i;
	
	//filling the 2D array with the values of the Edges
	//creating a symetric matrix
	for(i = 0; i < numEdges; i++){
		int orig = (theGraph -> edges)[i].orig;
		int dest = (theGraph -> edges)[i].dest;
		int weight = (theGraph -> edges)[i].weight;
		theArray[orig][dest] = weight;
		theArray[dest][orig] = weight;
	}
	
	//because the weights of the connection Vi - Vi dont matter
	//the weight is more than the max
	for(i = 1; i <= numVert; i++){
		theArray[i][i] = MAX_WEIGHT + 1;
	}
	
	printGraph(theGraph);
	
	printf("==============================================\n");
	print2DArray(theArray,theGraph -> numVert);	
	
	
	printf("==============================================\n");
	printf("==============================================\n");
	printf("Doing Normal\n");
	
	normal(theArray,numVert);
	//normal(&a,n);
	//gpu(a,n);

  return 0;
}

