kruskal: kruskalGPU.cu
	nvcc -o kruskal  kruskalGPU.cu

clean:
	rm kruskal cpuResult gpuResult	
