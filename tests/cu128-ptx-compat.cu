// Exercise the actual shim primitives without thread-block cluster launches.
#include <cuda/ptx>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include "cu128-ptx-compat.cuh"

#define CUDA_CHECK(call) do { \
  cudaError_t cuda_check_status = (call); \
  if (cuda_check_status != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(cuda_check_status)); \
    return EXIT_FAILURE; \
  } \
} while (0)

template <int Bytes>
__global__ void copy_with_shim(const unsigned char* input,
                               unsigned char* output, int* status) {
  __shared__ __align__(16) unsigned char staging[Bytes];
  __shared__ __align__(8) uint64_t barrier;
  cuda::ptx::mbarrier_init(&barrier, 1);
  asm volatile("fence.proxy.async.shared::cta;" : : : "memory");
  vllm::cu128_ptx::mbarrier_arrive_expect_tx(&barrier, Bytes);
  vllm::cu128_ptx::cp_async_bulk(staging, input, Bytes, &barrier);
  const auto started = clock64();
  while (!vllm::cu128_ptx::mbarrier_try_wait_parity(&barrier, 0)) {
    if (clock64() - started > 100000000ULL) {
      *status = -1;
      return;
    }
  }
  asm volatile("fence.proxy.async.shared::cta;" : : : "memory");
  for (int i = 0; i < Bytes; ++i) output[i] = staging[i];
  const auto address = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));
  asm volatile("mbarrier.inval.shared::cta.b64 [%0];" : : "r"(address) : "memory");
  *status = 1;
}

int main() {
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
  unsigned char *input, *output;
  int* status;
  CUDA_CHECK(cudaMalloc(&input, 128));
  CUDA_CHECK(cudaMalloc(&output, 128));
  CUDA_CHECK(cudaMalloc(&status, sizeof(int)));
  for (int iteration = 0; iteration < 8; ++iteration) {
    unsigned char expected[128], actual[128];
    for (int i = 0; i < 128; ++i) expected[i] = (i * 7 + iteration * 13) % 256;
    CUDA_CHECK(cudaMemcpy(input, expected, 128, cudaMemcpyHostToDevice));
    for (int bytes : {16, 128}) {
      CUDA_CHECK(cudaMemset(output, 0, 128));
      CUDA_CHECK(cudaMemset(status, 0, sizeof(int)));
      if (bytes == 16) copy_with_shim<16><<<1, 1>>>(input, output, status);
      else copy_with_shim<128><<<1, 1>>>(input, output, status);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      int result = 0;
      CUDA_CHECK(cudaMemcpy(&result, status, sizeof(int), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(actual, output, bytes, cudaMemcpyDeviceToHost));
      if (result != 1) {
        std::fprintf(stderr, "Barrier did not complete (status=%d)\n", result);
        return EXIT_FAILURE;
      }
      for (int i = 0; i < bytes; ++i) {
        if (actual[i] != expected[i]) {
          std::fprintf(stderr, "Copy mismatch: iteration=%d bytes=%d index=%d\n", iteration, bytes, i);
          return EXIT_FAILURE;
        }
      }
    }
  }
  CUDA_CHECK(cudaFree(status));
  CUDA_CHECK(cudaFree(output));
  CUDA_CHECK(cudaFree(input));
  std::printf("PASS: CUDA 12.8 PTX shim on %s (SM %d.%d), 16/128 bytes x 8 iterations\n",
              properties.name, properties.major, properties.minor);
  return EXIT_SUCCESS;
}
