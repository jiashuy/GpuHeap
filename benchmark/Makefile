INCLUDES = -I../src/include \
					 -I../third_party/cuCollections/include \
           -I../third_party/bgpq/heap
					
# 					 -I../third_party/priority_queue
NVCC = /usr/local/cuda-12.6/bin/nvcc

NVCC_FLAGS = $(INCLUDES) -std=c++17 -O2 -arch=sm_90


benchmark: benchmark.cu ../src/include/detail/gpu_heap_kernels.cuh
	$(NVCC) $(NVCC_FLAGS)  benchmark.cu -o benchmark

clean:
	rm benchmark