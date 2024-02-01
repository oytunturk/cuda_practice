#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void unique_idx_calc(int *input) {
  int x_offset = blockDim.x*blockIdx.x;
  int y_offset = blockDim.x*gridDim.x*blockIdx.y;
  int tid = threadIdx.x + x_offset + y_offset;
  printf("x_offset=%d y_offset=%d %d input[%d]=%d\n", x_offset, y_offset, tid, tid, input[tid]);
}

int main() {
  int arr[] = {10, 20, 30, 40, 50, 60, 70, 80,\
               90, 100, 110, 120, 130, 140, 150, 160};
  int *arr_d = NULL;
  int arr_size = 16;
  //2-D grid (2x2x1), each block is 4x1x1
  dim3 block(4, 1, 1);
  dim3 grid(2, 2, 1);

  size_t arr_byte_size = arr_size * sizeof(int);
  cudaMalloc((void**)(&arr_d), arr_byte_size);
  cudaMemcpy(arr_d, arr, arr_byte_size, cudaMemcpyHostToDevice);

  unique_idx_calc << <grid, block>> > (arr_d);

  //Wait until kernel call is completed
  cudaDeviceSynchronize();

  cudaFree((void*)arr_d);

  //Reset the device
  cudaDeviceReset();

  return 0;
}