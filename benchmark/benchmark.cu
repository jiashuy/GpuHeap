#include "gpu_heap.cuh"
#include <cuco/pair.cuh>
#include <iostream>

#include <thrust/device_vector.h>

#include <cstdint>
#include <random>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)


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

    

    GpuHeap<pair<Key, Value>, pair_less<pair<Key, Value>>> pq(Capacity);
    CUDA_CHECK(cudaDeviceSynchronize()); std::cout << __LINE__ << "\n";


    std::vector<pair<Key, Value>> h_pairs(NumKeys);
    generate_kv_pairs_uniform<Key, Value>(h_pairs.begin(), h_pairs.end());
    const thrust::device_vector<pair<Key, Value>> d_pairs(h_pairs);
    CUDA_CHECK(cudaDeviceSynchronize()); std::cout << __LINE__ << "\n";

    
    pq.push(d_pairs.begin(), d_pairs.end());
    CUDA_CHECK(cudaDeviceSynchronize()); std::cout << __LINE__ << "\n";
  
}

template <typename Key, typename Value, int Capacity, int NumKeys>
static void BM_delete()
{
  
    

    GpuHeap<pair<Key, Value>, pair_less<pair<Key, Value>>> pq(Capacity);

    std::vector<pair<Key, Value>> h_pairs(NumKeys);
    generate_kv_pairs_uniform<Key, Value>(h_pairs.begin(), h_pairs.end());
    thrust::device_vector<pair<Key, Value>> d_pairs(h_pairs);

    pq.push(d_pairs.begin(), d_pairs.end());
    cudaDeviceSynchronize();

    
    pq.pop(d_pairs.begin(), d_pairs.end());
    cudaDeviceSynchronize();
  
}


int main(int argc, char* argv[]) {

  BM_insert<int64_t, int64_t, 128 * 1024 * 1024, 1024 * 1024>();
  CUDA_CHECK(cudaDeviceSynchronize()); std::cout << __LINE__ << "\n";
  BM_delete<int64_t, int64_t, 128 * 1024 * 1024, 1024 * 1024>();

}