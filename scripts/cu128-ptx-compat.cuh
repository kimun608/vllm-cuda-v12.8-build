#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// CUDA 12.8 supports these PTX 8.6 instructions, but its bundled cuda::ptx
// headers do not expose all three overloads used by cooperative_topk.
// Keep the original relaxed CTA semantics and barrier/phase protocol.
// ISA reference: https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html
#if CUDART_VERSION >= 12080 && CUDART_VERSION < 12090
namespace vllm::cu128_ptx {
__device__ __forceinline__ bool mbarrier_try_wait_parity(uint64_t* barrier,
                                                       uint32_t parity) {
  const auto address = static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
  uint32_t complete;
  asm volatile(
      "{ .reg .pred P;\n"
      "mbarrier.try_wait.parity.relaxed.cta.shared::cta.b64 P, [%1], %2;\n"
      "selp.b32 %0, 1, 0, P; }"
      : "=r"(complete) : "r"(address), "r"(parity) : "memory");
  return complete != 0;
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* barrier,
                                                       uint32_t bytes) {
  const auto address = static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
  asm volatile(
      "mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 _, [%0], %1;"
      : : "r"(address), "r"(bytes) : "memory");
}

__device__ __forceinline__ void cp_async_bulk(void* destination,
                                            const void* source, uint32_t bytes,
                                            uint64_t* barrier) {
  const auto destination_address =
      static_cast<uint32_t>(__cvta_generic_to_shared(destination));
  const auto source_address = __cvta_generic_to_global(source);
  const auto barrier_address =
      static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
      "[%0], [%1], %2, [%3];"
      : : "r"(destination_address), "l"(source_address), "r"(bytes),
          "r"(barrier_address) : "memory");
}
}  // namespace vllm::cu128_ptx
#endif
