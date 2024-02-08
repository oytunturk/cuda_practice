#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void not_divergent() {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = threadIdx.x / 32;
  float a, b;
  if (warp_id % 2) {
      a=100.0;
      b=50.0;
  } else {
      a=200.0;
      b=75.0;
  }
}

__global__ void divergent() {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  float a, b;
  if (gid % 2) {
      a=100.0;
      b=50.0;
  } else {
      a=200.0;
      b=75.0;
  }
}


int main() {
  int size = 1 << 22;
  dim3 block(128);
  dim3 grid((size + block.x - 1)/block.x);

  not_divergent << <grid, block>> > ();
  //divergent << <grid, block>> > ();

  //Wait until kernel call is completed
  cudaDeviceSynchronize();

  //Reset the device
  cudaDeviceReset();

  return 0;
}

//You can profile these two functions using: (assuming your output was set to hello_cuda in nvcc (i.e. you compiled using !nvcc -G hello_cuda.cu -o hello_cuda)
// -G turns off compiler optimizations. Still some optimizations are happening automatically so your branch efficiency will not be as worse as the worst case.
//!nvprof --metrics branch_efficiency hello_cuda