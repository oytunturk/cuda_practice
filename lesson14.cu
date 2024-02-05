#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>

__global__ void sum_array_gpu_1d(int *a, int *b, int *result, size_t len) {
  int gid = blockDim.x * blockIdx.x + threadIdx.x;

  if (gid < len) {
    //printf("gid = %d\n", gid);
    result[gid] = a[gid] + b[gid];
  }
}

int main() {

  int *h_arr1 = NULL;
  int *d_arr1 = NULL;
  int *h_arr2 = NULL;
  int *d_arr2 = NULL;
  int *h_gpu_result = NULL;
  int *d_gpu_result = NULL;
  int *h_cpu_result = NULL;
  
  int i; 
  int len = 100;

  //3-D grid is 4x4x4, each block is 2x2x2
  dim3 block(16, 1, 1);
  dim3 grid((len/block.x)+1, 1, 1);

  size_t byte_size = len * sizeof(int);
  h_arr1 = (int*)malloc(byte_size);
  h_arr2 = (int*)malloc(byte_size);
  h_gpu_result = (int*)malloc(byte_size);
  h_cpu_result = (int*)malloc(byte_size);
  for (i=0; i<len; i++) h_arr1[i]= (i*10);
  for (i=0; i<len; i++) h_arr2[i]= (i*5);
  cudaMalloc((void**)(&d_arr1), byte_size);
  cudaMalloc((void**)(&d_arr2), byte_size);
  cudaMalloc((void**)(&d_gpu_result), byte_size);
  cudaMemcpy(d_arr1, h_arr1, byte_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_arr2, h_arr2, byte_size, cudaMemcpyHostToDevice);

  sum_array_gpu_1d << <grid, block>> > (d_arr1, d_arr2, d_gpu_result, len);

  //Wait until kernel call is completed
  cudaDeviceSynchronize();

  cudaMemcpy(h_gpu_result, d_gpu_result, byte_size, cudaMemcpyDeviceToHost);

  //Also do it on cpu to verify results
  for(i=0; i<len; i++) {
    h_cpu_result[i] = h_arr1[i] + h_arr2[i];
  }
  int err = 0;
  for(i=0; i<len; i++) {
    if (fabs(h_cpu_result[i]-h_gpu_result[i])>1.0e-10) {
      printf("ERROR! h_cpu_result[%d] = %d --> h_gpu_result[%d] = h_arr1[%d] + h_arr2[%d] = %d + %d = %d\n", i, h_cpu_result[i], i, i, i, h_arr1[i], h_arr2[i], h_gpu_result[i]);
    }
  }
  if (err==0) {
    printf("Success! All gpu results match cpu results\n");
  }
  cudaFree((void*)d_arr1);
  cudaFree((void*)d_arr2);
  cudaFree((void*)d_gpu_result);
  free(h_arr1);
  free(h_arr2);
  free(h_gpu_result);
  free(h_cpu_result);

  //Reset the device
  cudaDeviceReset();

  return 0;
}