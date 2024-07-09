#include <cstdio>
#include <cuda_runtime.h>

__global__ void hello() {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i = 0; i < 8; i++) {
    if (tid == i) {
      printf("Hello from thread %d\n", tid);
    }
    __syncthreads();
  }
}

int main() {
  int nthreads = 8;
  int nblocks = 1;
  hello<<<nblocks, nthreads>>>();
  cudaDeviceSynchronize();
  return 0;
}