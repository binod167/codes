#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define N 40

__global__ void matrixMul(int *a, int *b, int *c)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  int sum = 0;
  for (int i = 0; i < N; ++i)
  {
    sum += a[row * N + i] * b[i * N + col];
  }

  c[row * N + col] = sum;
}

void matrixMulCPU(int *a, int *b, int *c)
{
  for (int row = 0; row < N; ++row)
  {
    for (int col = 0; col < N; ++col)
    {
      int sum = 0;
      for (int i = 0; i < N; ++i)
      {
        sum += a[row * N + i] * b[i * N + col];
      }
      c[row * N + col] = sum;
    }
  }
}

double getSeconds()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

int main()
{
  int a[N][N], b[N][N], c[N][N], c_CPU[N][N];
  int *dev_a, *dev_b, *dev_c;

  // Initialize matrices a and b
  for (int i = 0; i <= N; ++i)
  {
    for (int j = 0; j <= N; ++j)
    {
      a[i][j] = i + j + 1;
      b[i][j] = i * j + 1;
    }
  }

  // Allocate memory on the device
  cudaMalloc((void **)&dev_a, N * N * sizeof(int));
  cudaMalloc((void **)&dev_b, N * N * sizeof(int));
  cudaMalloc((void **)&dev_c, N * N * sizeof(int));

  // Copy matrices a and b from host to device
  cudaMemcpy(dev_a, a, N * N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, N * N * sizeof(int), cudaMemcpyHostToDevice);

  // Define grid and block dimensions
  dim3 threadsPerBlock(N, N);
  dim3 blocksPerGrid(1, 1);

  // Measure GPU execution time
  double startGPU = getSeconds();

  // Launch the matrix multiplication kernel
  matrixMul<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_c);

  // Copy result matrix c from device to host
  cudaMemcpy(c, dev_c, N * N * sizeof(int), cudaMemcpyDeviceToHost);

  double endGPU = getSeconds();
  double timeGPU = endGPU - startGPU;

  // Print the result matrix
  printf("GPU Result Matrix:\n");
  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      printf("%d ", c[i][j]);
    }
    printf("\n");
  }

  // Measure CPU execution time
  double startCPU = getSeconds();

  // Perform matrix multiplication on the CPU
  matrixMulCPU((int *)a, (int *)b, (int *)c_CPU);

  double endCPU = getSeconds();
  double timeCPU = endCPU - startCPU;

  // Print the result matrix
  printf("\nCPU Result Matrix:\n");
  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      printf("%d ", c_CPU[i][j]);
    }
    printf("\n");
  }

  // Calculate speedup
  double speedup = timeCPU / timeGPU;

  // Print execution times and speedup
  printf("\nExecution Time (GPU): %.6f seconds\n", timeGPU);
  printf("Execution Time (CPU): %.6f seconds\n", timeCPU);
  printf("Speedup: %.2f\n", speedup);

  // Free device memory
  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);

  return 0;
}