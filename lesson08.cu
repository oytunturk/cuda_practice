#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void print_details() {
  printf("threadIdx.x=%d, threadIdx.y=%d, threadIdx.z=%d,\
  blockIdx.x=%d, blockIdx.y=%d, blockIdx.z=%d,\
  blockDim.x=%d, blockDim.y=%d, blockDim.z=%d,\
  gridDim.x=%d, gridDim.y=%d, gridDim.z=%d\n",\
  threadIdx.x, threadIdx.y, threadIdx.z,\
  blockIdx.x, blockIdx.y, blockIdx.z,\
  blockDim.x, blockDim.y, blockDim.z,\
  gridDim.x, gridDim.y, gridDim.z);
}

int main() {
  int nx = 8;
  int ny = 8;
  int nz = 8;
  //2-D grid (4x4x4), each block is 2x2x2
  dim3 block(2, 2, 2);
  dim3 grid(nx / block.x, ny / block.y, nz/block.z);

  print_details << <grid, block>> > ();

  //Wait until kernel call is completed
  cudaDeviceSynchronize();

  //Reset the device
  cudaDeviceReset();

  return 0;
}