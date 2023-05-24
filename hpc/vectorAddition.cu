% % cu

#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <chrono>

        // CUDA kernel for vector addition
        __global__ void
        vectorAdd(float *a, float *b, float *c, int n)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n)
  {
    c[i] = a[i] + b[i];
  }
}

// Function to initialize vectors with random values
void initializeVectors(float *a, float *b, int n)
{
  for (int i = 0; i < n; ++i)
  {
    a[i] = static_cast<float>(rand()) / RAND_MAX;
    b[i] = static_cast<float>(rand()) / RAND_MAX;
  }
}

int main()
{
  int n = 1000000; // Size of the vectors

  // Allocate memory for host vectors
  float *h_a = new float[n];
  float *h_b = new float[n];
  float *h_c = new float[n];

  // Initialize host vectors
  initializeVectors(h_a, h_b, n);

  // Allocate memory for device vectors
  float *d_a, *d_b, *d_c;
  cudaMalloc((void **)&d_a, n * sizeof(float));
  cudaMalloc((void **)&d_b, n * sizeof(float));
  cudaMalloc((void **)&d_c, n * sizeof(float));

  // Copy host vectors to device
  cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);

  // Define grid and block sizes
  int blockSize = 256;
  int numBlocks = (n + blockSize - 1) / blockSize;

  // Create CUDA events to measure execution time
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Record start event
  cudaEventRecord(start);

  // Launch the CUDA kernel
  vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);

  // Record stop event
  cudaEventRecord(stop);

  // Synchronize the device
  cudaDeviceSynchronize();

  // Calculate the elapsed GPU time
  float gpuMilliseconds = 0;
  cudaEventElapsedTime(&gpuMilliseconds, start, stop);
  float gpuTime = gpuMilliseconds / 1000.0; // Convert to seconds

  // Measure CPU time using std::chrono
  auto cpuStartTime = std::chrono::high_resolution_clock::now();

  // Perform vector addition on the CPU
  for (int i = 0; i < n; ++i)
  {
    h_c[i] = h_a[i] + h_b[i];
  }

  // Measure CPU time again and calculate elapsed time
  auto cpuEndTime = std::chrono::high_resolution_clock::now();
  float cpuTime = std::chrono::duration_cast<std::chrono::milliseconds>(cpuEndTime - cpuStartTime).count() / 1000.0;

  // Calculate the speedup
  float speedup = cpuTime / gpuTime;

  // Display the speedup
  std::cout << "CPU Time: " << cpuTime << std::endl;
  std::cout << "GPU Time: " << gpuTime << std::endl;
  std::cout << "Speedup: " << speedup << std::endl;

  // Free device memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  // Free host memory
  delete[] h_a;
  delete[] h_b;
  delete[] h_c;

  return 0;
}