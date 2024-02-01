#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

//This can no longer access all elements in input array
__global__ void unique_idx_calc_threadIdx(int *input) {
  int tid = threadIdx.x;
  printf("input[%d]=%d\n", tid, input[tid]);
}

//Access all elements using threadIdx, blockDim and blockIdx
__global__ void unique_idx_calc_threadIdx_blockDim(int *input) {
  int offset = blockDim.x * blockIdx.x;
  int tid = offset + threadIdx.x;
  printf("input[%d]=%d\n", tid, input[tid]);
}

int main() {
  int arr[] = {10, 20, 30, 40, 50, 60, 70, 80};
  int *arr_d = NULL;
  int arr_size = 8;
  int nx = arr_size;
  int ny = 1;
  int nz = 1;
  //2-D grid (4x4x4), each block is 2x2x2
  dim3 block(4, 1, 1);
  dim3 grid(nx / block.x, ny / block.y, nz/block.z);

  size_t arr_byte_size = arr_size * sizeof(int);
  cudaMalloc((void**)(&arr_d), arr_byte_size);
  cudaMemcpy(arr_d, arr, arr_byte_size, cudaMemcpyHostToDevice);

  //unique_idx_calc_threadIdx << <grid, block>> > (arr_d);
  unique_idx_calc_threadIdx_blockDim << <grid, block>> > (arr_d);

  //Wait until kernel call is completed
  cudaDeviceSynchronize();

  cudaFree((void*)arr_d);

  //Reset the device
  cudaDeviceReset();



  return 0;
}