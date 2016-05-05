#include <emmintrin.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

int find(int);
int uni(int,int);

int getWeight(int array[],int row, int col, int n);
int setWeight(int* array[],int row, int col, int n, int value);

#define MAX_WEIGHT  100
#define MAX_VERTICES  1000

int parent[MAX_VERTICES];

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

void gpu(Graph ** graph) {
	int numVert = (*graph) -> numVert;
	int numEdges = (*graph) -> numEdges;
	int* unionMatrix;
	
	int* checkArray;
	size_t pitch;
	
	int* d_weights;
	int* h_weights = (int*)malloc(numEdges * sizeof(int));
	Edge* h_edges = (Edge*)malloc(numEdges* sizeof(Edge));
	Edge* d_edges;
	Edge* d_resultEdges;
	//int N = 10;
	int* vertList;

    // starttime();
	cudaMalloc(&d_edges, numEdges * sizeof(Edge));
	cudaMalloc(&d_resultEdges, (numVert - 1) * sizeof(Edge));
	cudaMalloc(&d_weights, numEdges * sizeof(int));
	
	cudaMalloc(&vertList,2 * sizeof(int));

	cudaMallocPitch(&unionMatrix, &pitch,
                (numVert + 1) * sizeof(Edge), numVert + 1);
	cudaMalloc(&checkArray, (numVert+1)*sizeof(int));

	cudaMemcpy(d_edges, (*graph) -> edges, numEdges * sizeof(Edge), cudaMemcpyHostToDevice);
	cudaMemcpy(d_weights, ((*graph) -> weights) -> weights , numEdges * sizeof(int), cudaMemcpyHostToDevice);

	//printf("Data transfered!\n");

	thrust::device_ptr<Edge> t_edges(d_edges);
	thrust::device_ptr<int> t_weights(d_weights);




	thrust::sort_by_key( t_weights , t_weights + numEdges, t_edges);
	
	//endtime("end of sort");

	int numThreads = 16;
	int numBlocks = numVert / numThreads + 1;
	
	int j; 
	typeof(devFound) found;
	
	//__device__ int devFound;
	
	//cudaMemcpyToSymbol(devFound,&value,sizeof(int));
	dim3 threadsPerBlock(4,4);
	dim3 numBlocks2D(numVert/threadsPerBlock.x + 1,numVert/threadsPerBlock.y + 1);
		
	//printf("Number of Blocks %d\n",numBlocks2D.x);	
	//starttime();
	initializeUnionMatrix<<<numBlocks2D,threadsPerBlock>>>(unionMatrix,pitch,numVert);	
	//printGPUMatrix<<<1,1>>>(unionMatrix,numVert + 1);	
	//endtime("matrix initized");
	//cudaMemcpyToSymbol(devFound, found,sizeof(int));
	int counter = 0;
	
	for(j = 0;j < numEdges && counter < numVert ;j++){
		getValue<<<1,1>>>(d_edges,j,vertList);

		//starttime();

		checkSet<<<numBlocks,numThreads>>>(unionMatrix,checkArray,pitch,numVert,vertList);
		 
		//endtime(" check set multi"); 
		
		/***************************************************************************************
		* Inserting the node after it was checked that it didnt exist
		****************************************************************************************/
		//reset<<<1,1>>>();
		//starttime();
		arrayCheck<<<1,1>>>( checkArray,numVert);
		//endtime(" array check");
		
		//starttime();
        cudaMemcpyFromSymbol(&found, devFound, sizeof(found), 0, cudaMemcpyDeviceToHost);
		//endtime("cuda memcpy from symbol");

		if(found == 0){
			
			//starttime();
			insertResultingEdge<<<1,1>>>(d_edges,d_resultEdges,counter,j);
		    //endtime("insert resulting edge");	

			counter++;
			int numThreads_or = threadsPerBlock.x;
			int numBlocks_or = numBlocks2D.x;
		    
		    //starttime();
			setValue<<<1,1>>>(unionMatrix,vertList, pitch);	
		    //endtime(" set value function");
			//updating the rest of the sets

			//starttime();
			orCol<<<numBlocks_or,numThreads_or>>>(unionMatrix,pitch,numVert,vertList);
		    //endtime(" orCol function ");
			//printGPUMatrix<<<1,1>>>(unionMatrix,numVert);	
            
            //starttime();
			update<<<numBlocks,threadsPerBlock>>>(unionMatrix,pitch,numVert,vertList);	
			//endtime(" update fuction");
		}


		
	}
	
	//starttime();
	printEdges<<<1,1>>>(d_resultEdges,numVert - 1);
	//endtime(" print edges ");

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
			ne++;
			//printf("%d edge (%d,%d) =%d\n",ne++,a,b,min);
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
	
	Weights* weightsArray;	

	weightsArray = (Weights*)malloc( sizeof(Weights) + numEdges*sizeof(int));

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
			int temp = (rand() % MAX_WEIGHT) + 1;
			edge -> weight = temp;
			(graph -> edges)[edgeNumber] = (*edge);
			(weightsArray -> weights)[edgeNumber] = temp ;
			edgeNumber++;
		}
	}
	
	graph -> weights = weightsArray;
	
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
	theGraph = genGraph(5000,(unsigned) time(&t));

	/*
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
	for(i = 1; i < numVert; i++){
		theArray[i][i] = MAX_WEIGHT + 1;
	}
	*/
	//printGraph(theGraph);
	
	//printf("==============================================\n");
	//print2DArray(theArray,theGraph -> numVert);	
	
	
	printf("==============================================\n");
	printf("%s\n", "   serial version  ");
	printf("==============================================\n");
	//printf("Doing Normal\n");


	//starttime();
	//normal(theArray,numVert);
	//endtime("serial version  ");

	//normal(&a,n);
	//

	//gpu(a,n);
    printf("%s\n", "before kru's alg");
	//printGraph(theGraph);
	
	printf("==============================================\n");
	printf("%s\n","        parallel algorithm    " );
	printf("==============================================\n");
	printf("Doing Kru on GPU\n");

	starttime();
	gpu(&theGraph);
	endtime(" parallel algorithm ");

	//printGraph(theGraph);
	
  return 0;
}

