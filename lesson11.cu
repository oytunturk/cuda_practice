#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void unique_gid_calc_2d(int *input) {
  int threads_per_block = blockDim.x * blockDim.y; 
  int gid = threadIdx.x * blockDim.y + threadIdx.y + threads_per_block * (blockIdx.x * gridDim.y + blockIdx.y);
  printf("%d input[%d]=%d\n", gid, gid, input[gid]);
}

int main() {
  int arr[] = {10, 20, 30, 40, 50, 60, 70, 80,\
               90, 100, 110, 120, 130, 140, 150, 160};
  int *arr_d = NULL;
  int arr_size = 16;
  //2-D grid is 2x2x1, each block is 2x2x1
  dim3 block(2, 2, 1);
  dim3 grid(2, 2, 1);

  size_t arr_byte_size = arr_size * sizeof(int);
  cudaMalloc((void**)(&arr_d), arr_byte_size);
  cudaMemcpy(arr_d, arr, arr_byte_size, cudaMemcpyHostToDevice);

  unique_gid_calc_2d << <grid, block>> > (arr_d);

  //Wait until kernel call is completed
  cudaDeviceSynchronize();

  cudaFree((void*)arr_d);

  //Reset the device
  cudaDeviceReset();

  return 0;
}