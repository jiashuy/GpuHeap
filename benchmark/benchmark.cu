#include "gpu_heap.cuh"
#include <cuco/pair.cuh>
#include <iostream>

#include <thrust/device_vector.h>

#include <cstdint>
#include <random>
#include <vector>
#include "benchmark_utils.cuh"

__global__ void warmup_kernel(float *data, int64_t len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < len) {
      data[tid] += 1.0f;
    }
    
}


void warmup_GPU() {
float *d_data;
int64_t len = 1024L * 1024 * 1024;
cudaMalloc(&d_data, len * sizeof(float));
auto timer = Timer<float>(TimeUnit::MilliSecond);
for (int i = 0; i < 10; i++) {
  timer.start();
  warmup_kernel<<<(len + 1023)/1024, 1024>>>(d_data, len);
  timer.end();
  CUDA_CHECK(cudaDeviceSynchronize()); 
  std::cout << "Iteration " << i << " latency: " << timer.getResult() << " ms\n";
}

cudaDeviceSynchronize();
cudaFree(d_data);
}

using namespace cuco;

template <typename T>
struct pair_less {
  __host__ __device__ bool operator()(const T& a, const T& b) const { return a.first < b.first; }
};

template <typename Key, typename Value, typename OutputIt>
static void generate_kv_pairs_uniform(OutputIt output_begin, OutputIt output_end)
{
  std::random_device rd;
  std::mt19937 gen{rd()};

  const auto num_keys = std::distance(output_begin, output_end);

  for (auto i = 0; i < num_keys; ++i) {
    output_begin[i] = {static_cast<Key>(gen()), static_cast<Value>(gen())};
  }
}

template <typename Key, typename Value, int Capacity, int NumKeys>
static void BM_insert()
{

    auto timer = Timer<float>(TimeUnit::MilliSecond);

    GpuHeap<pair<Key, Value>, pair_less<pair<Key, Value>>> pq(Capacity);
    CUDA_CHECK(cudaDeviceSynchronize()); std::cout << __LINE__ << "\n";


    std::vector<pair<Key, Value>> h_pairs(Capacity);
    generate_kv_pairs_uniform<Key, Value>(h_pairs.begin(), h_pairs.end());
    const thrust::device_vector<pair<Key, Value>> d_pairs(h_pairs);
    CUDA_CHECK(cudaDeviceSynchronize()); std::cout << __LINE__ << "\n";

    int i = 0;
    for (auto begin = d_pairs.begin(); begin != d_pairs.end(); begin+=NumKeys) {
      timer.start();
      pq.push(begin, begin + NumKeys);
      timer.end();
      CUDA_CHECK(cudaDeviceSynchronize()); 
      std::cout << "Iteration " << i++ << " latency: " << timer.getResult() << " ms\n";
    }

    std::cout << __LINE__ << "\n";
  
}

template <typename Key, typename Value, int Capacity, int NumKeys>
static void BM_delete()
{
  
    
    auto timer = Timer<float>(TimeUnit::MilliSecond);
    GpuHeap<pair<Key, Value>, pair_less<pair<Key, Value>>> pq(Capacity);

    std::vector<pair<Key, Value>> h_pairs(Capacity);
    generate_kv_pairs_uniform<Key, Value>(h_pairs.begin(), h_pairs.end());
    thrust::device_vector<pair<Key, Value>> d_pairs(h_pairs);
    for (auto begin = d_pairs.begin(); begin != d_pairs.end(); begin+=NumKeys) {
      pq.push(begin, begin + NumKeys);
    }
    CUDA_CHECK(cudaDeviceSynchronize()); 

    int i = 0;
    for (auto begin = d_pairs.begin(); begin != d_pairs.end(); begin+=NumKeys) {
      timer.start();
      pq.pop(begin, begin + NumKeys);
      timer.end();
      CUDA_CHECK(cudaDeviceSynchronize()); 
      std::cout << "Iteration " << i++ << " latency: " << timer.getResult() << " ms\n";
    }

    std::cout << __LINE__ << "\n";

    
    
    cudaDeviceSynchronize();
  
}


int main(int argc, char* argv[]) {

  warmup_GPU();

  BM_insert<int64_t, int64_t, 128 * 1024 * 1024, 1024 * 1024>();
  CUDA_CHECK(cudaDeviceSynchronize()); std::cout << __LINE__ << "\n";
  BM_delete<int64_t, int64_t, 128 * 1024 * 1024, 1024 * 1024>();

}