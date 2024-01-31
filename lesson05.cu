#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void hello_world() {
  printf("Hello CUDA\n");
}

int main() {
  //Runs it once (grid dimension x block dimension)
  //Note that both grid and block can be up to 3-dimensions
  //hello_world << <1, 1>> > ();
  //Runs it 20 times
  hello_world << <1, 20>> > ();

  //Wait until kernel call is completed
  cudaDeviceSynchronize();
  
  //Reset the device
  cudaDeviceReset();

  return 0;
}