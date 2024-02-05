#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void test_memory_transfer_3d(int *input, size_t len) {
  
  int num_threads_per_block = blockDim.x * blockDim.y * blockDim.z;
  // 0, 1, 2, ... index of thread inside block
  int ind_in_block =  blockDim.x * blockDim.y * threadIdx.z + threadIdx.x * blockDim.y + threadIdx.y;
  // 0, 1, 2, ... index of block inside grid 
  int ind_in_grid = gridDim.x * gridDim.y * blockIdx.z + blockIdx.x * gridDim.y + blockIdx.y;
  int gid = ind_in_block + ind_in_grid * num_threads_per_block;

  if (gid<len) {
    printf("tid=%d gid=%d value=%d\n", threadIdx.x, gid, input[gid]);
  }
}

int main() {

  int *h_arr = NULL;
  int *d_arr = NULL;
  int len = 64;
  //int len = 110;
  //3-D grid is 4x4x4, each block is 2x2x2
  dim3 block(2, 2, 2);
  dim3 grid(4, 4, 4);

  size_t byte_size = len * sizeof(int);
  h_arr = (int*)malloc(byte_size);
  for (int i=0; i<len; i++) h_arr[i]= (i*1);
  cudaMalloc((void**)(&d_arr), byte_size);
  cudaMemcpy(d_arr, h_arr, byte_size, cudaMemcpyHostToDevice);

  test_memory_transfer_3d << <grid, block>> > (d_arr, len);

  //Wait until kernel call is completed
  cudaDeviceSynchronize();

  cudaFree((void*)d_arr);
  free(h_arr);

  //Reset the device
  cudaDeviceReset();

  return 0;
}