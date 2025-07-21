#include <cassert>
#include <chrono>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <random>

// You may increase this value to test larger matrices
// But it will be slow on CPU
constexpr int MAXN = 2048;
constexpr int BSZ = 16;

/**
 * @brief A naive implementation of matrix multiplication on CPU.
 * Perform C = A * B, where A is M x K, B is K x N, and C is M x N.
 */
void naiveSgemm(float *a, float *b, float *c, const int M, const int N,
                const int K) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float sum = 0.0;
      for (int k = 0; k < K; ++k) {
        sum += a[m * K + k] * b[k * N + n];
      }
      c[m * N + n] = sum;
    }
  }
}

/**
 * @brief A naive implementation of matrix multiplication on GPU.
 * Perform C = A * B, where A is M x K, B is K x N, and C is M x N.
 */
__global__ void naiveSgemm2D(float *a, float *b, float *c, const int M,
                             const int N, const int K) {
  int m = blockIdx.x * blockDim.x + threadIdx.x; // Row index
  int n = blockIdx.y * blockDim.y + threadIdx.y; // Column index
  if (m < M && n < N) {
    float sum = 0.0;
    for (int k = 0; k < K; ++k) {
      sum += a[m * K + k] * b[k * N + n];
    }
    c[m * N + n] = sum;
  }
}

__global__ void mySgemm2D(float *a, float *b, float *c, const int M,
                             const int N, const int K) {
  int const m1 = blockIdx.x * BSZ;
  int const n1 = blockIdx.y * BSZ;
  int const m2 = threadIdx.x;
  int const n2 = threadIdx.y;
  for (int i = 0; i < K; i += BSZ) {
    __shared__ float Asub[BSZ * BSZ];
    __shared__ float Bsub[BSZ * BSZ];
    Asub[m2 * BSZ + n2] = a[(m1 + m2) * K + (i + n2)];
    // Bsub[m2 * BSZ + n2] = b[(i + m2) * N + (n1 + n2)];
    Bsub[n2 * BSZ + m2] = b[(i + m2) * N + (n1 + n2)];
    __syncthreads();
    float sum = 0.0;
    for (int j = 0; j < BSZ; j++) {
      // sum += Asub[m2 * BSZ + j] * Bsub[j * BSZ + n2];
      sum += Asub[m2 * BSZ + j] * Bsub[n2 * BSZ + j];
    }
    c[(m1 + m2) * N + (n1 + n2)] += sum;
    __syncthreads();
  }
}

void launchSgemm2D(float *a, float *b, float *c, const int M, const int N,
                   const int K) {
  dim3 block(BSZ, BSZ);
  dim3 grid(M / BSZ, N / BSZ); // assuming M, N, K are multiples of block size
  mySgemm2D<<<grid, block>>>(a, b, c, M, N, K);
}

void initialize(float *a, float *b, float *c, const int M, const int N,
                const int K) {
  auto gen = std::mt19937(2024);
  auto dis = std::uniform_real_distribution<float>(-1.0, 1.0);
  for (int i = 0; i < M * K; ++i) {
    a[i] = dis(gen);
  }
  for (int i = 0; i < K * N; ++i) {
    b[i] = dis(gen);
  }
  for (int i = 0; i < M * N; ++i) {
    c[i] = 0.0;
  }
}

/** 
 * @brief Launch sgemm using cuBLAS
 */
void launchCublasSgemm(float *a, float *b, float *c, const int M, const int N,
                       const int K) {
  cublasHandle_t handle;
  cublasCreate(&handle);
  float alpha = 1.0;
  float beta = 0.0;
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, b, N, a, K,
              &beta, c, N);
}

bool compare(float *a, float *b, const int N) {
  for (int i = 0; i < N; ++i) {
    if (std::abs(a[i] - b[i]) > 1e-3) {
      printf("Mismatch at index %d: %f vs %f\n", i, a[i], b[i]);
      return false;
    }
  }
  printf("Results match\n");
  return true;
}

int main() {
  float *a, *b, *c, *ans1, *ans2;
  a = new float[MAXN * MAXN];
  b = new float[MAXN * MAXN];
  c = new float[MAXN * MAXN];
  ans1 = new float[MAXN * MAXN];
  ans2 = new float[MAXN * MAXN];
  initialize(a, b, c, MAXN, MAXN, MAXN);
  
  decltype(std::chrono::high_resolution_clock::now()) start, end;
  std::chrono::duration<double> elapsed;

  // ********** CPU **********
  // start = std::chrono::high_resolution_clock::now();
  // naiveSgemm(a, b, c, MAXN, MAXN, MAXN);
  // end = std::chrono::high_resolution_clock::now();
  // elapsed = end - start;
  // printf("CPU time: %.3fs\n", elapsed.count());

  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, MAXN * MAXN * sizeof(float));
  cudaMalloc(&d_b, MAXN * MAXN * sizeof(float));
  cudaMalloc(&d_c, MAXN * MAXN * sizeof(float));
  cudaMemcpy(d_a, a, MAXN * MAXN * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, MAXN * MAXN * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, c, MAXN * MAXN * sizeof(float), cudaMemcpyHostToDevice);

  // ********** GPU **********
  start = std::chrono::high_resolution_clock::now();
  launchSgemm2D(d_a, d_b, d_c, MAXN, MAXN, MAXN);
  cudaDeviceSynchronize();
  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  printf("GPU time: %.3fs\n", elapsed.count());
  cudaMemcpy(ans1, d_c, MAXN * MAXN * sizeof(float), cudaMemcpyDeviceToHost);

  // ********** cuBLAS **********
  start = std::chrono::high_resolution_clock::now();
  launchCublasSgemm(d_a, d_b, d_c, MAXN, MAXN, MAXN);
  cudaDeviceSynchronize();
  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  printf("cuBLAS time: %.3fs\n", elapsed.count());
  cudaMemcpy(ans2, d_c, MAXN * MAXN * sizeof(float), cudaMemcpyDeviceToHost);

  compare(ans2, ans1, MAXN * MAXN);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}
