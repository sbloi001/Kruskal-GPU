#include <emmintrin.h>
#include <sys/time.h>
#include <stdio.h>

int find(int);
int uni(int,int);

int getWeight(int array[],int row, int col, int n);
int setWeight(int* array[],int row, int col, int n, int value);

int i,j,k,a,b,u,v,ne=1;
int min,mincost=0,parent[9];


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
__global__ void gpu_sqrt(float* a, int N) {
   //int element = blockIdx.x*blockDim.x + threadIdx.x;
   //if (element < N) a[element] = sqrt(a[element]);
}

void gpu(int* array[], int n) {
   //int numThreads = 1024;
   //int numBlocks = N / 1024 + 1;

   //float* gpuA;
   //cudaMalloc(&gpuA, N*sizeof(float));
   //cudaMemcpy(gpuA, a, N*sizeof(float), cudaMemcpyHostToDevice);
   //gpu_sqrt<<<numBlocks, numThreads>>>(gpuA, N);
   //cudaMemcpy(a, gpuA, N*sizeof(float), cudaMemcpyDeviceToHost);
   //cudaFree(&gpuA);
}
                                                                                                                                                                                               
 
 int getWeight(int array[],int row, int col, int n){
	 return array[row*n + col];
 }
 
 int setWeight(int* array[],int row, int col, int n, int value){
	 (*array)[row*n + col] = value;
 }

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

int main()                                                                                                                                                                                  
{           
	//dimension of the adjacency matrix nxn
	int n = 3;
	
	//size of the array;
	int size = n * n;
	
	//adjacency matrix in 1d array
	int array[size];
	
	init(&a,n);
	
	normal(&a,n);
	gpu(&a,n);

  return 0;
}

