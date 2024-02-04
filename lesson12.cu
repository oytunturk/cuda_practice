#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

//Doesn't check for size, so even the input is less than total threads possible by the provided grid and block dimensions, it will try to execute all
__global__ void test_memory_transfer(int *input) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  printf("tid=%d gid=%d value=%d\n", threadIdx.x, gid, input[gid]);
}

//Checks for size and doesn't execute if it's out of bounds
__global__ void test_memory_transfer2(int *input, size_t len) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid<len) {
    printf("tid=%d gid=%d value=%d\n", threadIdx.x, gid, input[gid]);
  }
}

int main() {

  int *h_arr = NULL;
  int *d_arr = NULL;
  //int len = 128;
  int len = 110;
  //2-D grid is 2x2x1, each block is 2x2x1
  dim3 block(64, 1, 1);
  dim3 grid(2, 1, 1);

  size_t byte_size = len * sizeof(int);
  h_arr = (int*)malloc(byte_size);
  for (int i=0; i<len; i++) h_arr[i]= (i*10);
  cudaMalloc((void**)(&d_arr), byte_size);
  cudaMemcpy(d_arr, h_arr, byte_size, cudaMemcpyHostToDevice);

  //test_memory_transfer << <grid, block>> > (d_arr);
  test_memory_transfer2 << <grid, block>> > (d_arr, len);

  //Wait until kernel call is completed
  cudaDeviceSynchronize();

  cudaFree((void*)d_arr);
  free(h_arr);

  //Reset the device
  cudaDeviceReset();

  return 0;
}