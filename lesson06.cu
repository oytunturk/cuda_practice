#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void hello_world() {
  printf("Hello CUDA\n");
}

int main() {
  int nx = 16;
  int ny = 4;
  //2-D grid (2x2), each block is also 2-D 8x2
  dim3 block(8, 2, 1);
  dim3 grid(nx / block.x, ny / block.y);
  
  hello_world << <grid, block>> > ();

  //Wait until kernel call is completed
  cudaDeviceSynchronize();
  
  //Reset the device
  cudaDeviceReset();

  return 0;
}