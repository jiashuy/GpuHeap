INCLUDES = -I../src/include \
           -I../third_party/bgpq/heap \
					 -I../third_party/priority_queue
NVCC = /usr/local/cuda-12.6/bin/nvcc

NVCC_FLAGS = $(INCLUDES) -std=c++11 -Wall -O2 -arch=sm_90


benchmark: benchmark.cu
	$(NVCC) $(NVCC_FLAGS) -c benchmark.cu -o benchmark