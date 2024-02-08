#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void print_warp_details() {
  int gid = blockIdx.y * gridDim.x * blockDim.x + \
            blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = threadIdx.x / 32;
  int global_block_id = blockIdx.y * gridDim.x + blockIdx.x;

  printf("threadIdx.x=%d blockIdx.x=%d blockIdx.y=%d gid=%d warp_id=%d global_block_id=%d\n",\
          threadIdx.x, blockIdx.x, blockIdx.y, gid, warp_id, global_block_id);
}

int main() {

  dim3 block(42);
  dim3 grid(2, 2);

  print_warp_details << <grid, block>> > ();

  //Wait until kernel call is completed
  cudaDeviceSynchronize();

  //Reset the device
  cudaDeviceReset();

  return 0;
}