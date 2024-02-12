#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void occupancy_count(int *results) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int x1 = 1;
  int x2 = 2;
  int x3 = 3;
  int x4 = 4;
  int x5 = 5;
  int x6 = 6;
  int x7 = 7;
  int x8 = 8;
  results[gid] = x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8; 
}

//!nvcc --ptxas-options=-v -o hello_cuda.out hello_cuda.cu
//will show how many registers are used
int main() {
  int arr_size = 1 << 22;
  int *arr_d = NULL;
  dim3 block(128);
  dim3 grid((arr_size + block.x - 1)/block.x);

  size_t arr_byte_size = arr_size * sizeof(int);
  cudaMalloc((void**)(&arr_d), arr_byte_size);

  occupancy_count << <grid, block>> > (arr_d);

  //Wait until kernel call is completed
  cudaDeviceSynchronize();

  //Reset the device
  cudaDeviceReset();

  cudaFree((void**)arr_d);

  return 0;
}