#include <emmintrin.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>

int find(int);
int uni(int,int);

int getWeight(int array[],int row, int col, int n);
int setWeight(int* array[],int row, int col, int n, int value);

/*
int i,j,k,a,b,u,v,ne=1;
int min,mincost=0,parent[9];
*/
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
	int startingPos = threadIdx.x * size * 3;
	int finalPos = startingPos + size * 3;
   //int element = blockIdx.x*blockDim.x + threadIdx.x;
   //if (element < N) a[element] = sqrt(a[element]);
}

void gpu(Graph ** graph) {
	char* allocatedVertices = (char*)malloc(sizeof(char)* ((*graph) -> numVert));
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
/*
void normal(int* array[], int n)
{	
	printf("\n\tImplementation of Kruskal's algorithm non parallelized\n");
	
	starttime();
	while(ne < n)
	{
		for(i=1,min=999;i<=n;i++)
		{
			for(j=1;j <= n;j++)
			{
				if(getWeight((*array),i,j,n) < min)
				{
					min=getWeight((*array),i,j,n);
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
		setWeight(array,i,j,n,999);
		setWeight(array,j,i,n,999);
	}
	printf("\n\tMinimum cost = %d\n",mincost);
	
	endtime("CPU");
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
*/

/*
	This generates a graph randomly. The array goes in the format ||V1|V2|weight||V2|V3|weight||...||
*/
Graph* genGraph(int numVert,unsigned int seed, int maxWeight){
	int numEdges = ((numVert * (numVert - 1))/2);
	Graph *graph;
	graph = (Graph*)malloc(sizeof(Graph) +  numEdges*sizeof(Edge));
	
	graph -> numEdges = numEdges;
	graph -> numVert = numVert;
	
	int i,j,edgeNumber = 0;
	
	//generating seed
	srand(seed);
	
	for(i = 0; i < numVert - 1; i++){
		for(j = i + 1; j< numVert;j++){
			Edge* edge = (Edge*)malloc(sizeof(Edge));
			edge -> orig = i;
			edge -> dest = j;
			edge -> weight = rand() % maxWeight;
			(graph -> edges)[edgeNumber] = (*edge);
			edgeNumber++;
		}
	}
	
	/*
	//assuring a complete graph
	for(i = 0; i < numVert - 1; i++){
		array[(i*3) + 0] = i;
		array[(i*3) + 1] = i+1;
		array[(i*3) + 2] = rand() % maxWeight;
	}
	
	int firstVert, secondVert;
	
	//randomly inserting edges
	for( j = i; j < numEdges; j++){
		firstVert = rand() % numVert;
		array[(j*3) + 0] = firstVert;
		
		while((secondVert = rand() % numVert) == firstVert);
			
		array[(j*3) + 1] = secondVert;
		array[(j*3) + 2] = rand() % maxWeight;
	}
	*/
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

void dimensioning(Graph* graph, int* array[][],int maxWeight){
	int numEdges = graph -> numEdges;
	
	int i;
	
	for(i = 0; i < numEdges; i++){
		int orig = (graph -> edges)[i].orig;
		int dest = (graph -> edges)[i].dest;
		int weight = (graph -> edges)[i].weight;
		(*array)[orig][dest] = weight;
		(*array)[dest][orig] = weight;
		(*array)[orig][orig] = maxWeight + 1;
		
	}
}

void print2DArray(int array[][], int numVert){
	int i,j;
	
	for(i = 0; i < numVert; i++){
		for(j = 0; j < numVert; j++){
			printf("%d-%d: %d\n",i,j,(array)[i][j]);
		}
	}
}

int main()                                                                                                                                                                                  
{         
	time_t t;
	Graph* theGraph;
	theGraph = genGraph(5,(unsigned) time(&t),100);
	
	int theArray[theGraph -> numVert][theGraph -> numVert];
	dimensioning(theGraph,&theArray,100);
	
	printGraph(theGraph);
	
	printf("==============================================\n");
	print2DArray(theArray,theGraph -> numVert);	
	
	//normal(&a,n);
	//gpu(a,n);

  return 0;
}

