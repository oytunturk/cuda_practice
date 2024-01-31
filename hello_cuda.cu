#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void hello_world() {
  printf("Hello CUDA\n");
}

int main() {
  hello_world << <1, 1>> > ();

  //Wait until kernel call is completed
  cudaDeviceSynchronize();
  
  //Reset the device
  cudaDeviceReset();

  return 0;
}