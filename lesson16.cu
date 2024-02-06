#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <time.h>

#include "my_tools.cuh"

//Add two arrays
//Also error handling
//Also measure time to compare cpu vs gpu processing and details of gpu operations (copying + kernel + copying back)
__global__ void sum_array_gpu_1d(int *a, int *b, int *result, size_t len) {
  int gid = blockDim.x * blockIdx.x + threadIdx.x;


  if (gid < len) {
    //printf("gid = %d\n", gid);
    result[gid] = a[gid] + b[gid];
  }
}

void sum_array_cpu_1d(int *a, int *b, int *result, int len) {
  for(int i=0; i<len; i++) {
    result[i] = a[i] + b[i];
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
  int len = 10000000;
  clock_t cpu_start, cpu_end;
  clock_t copy_start1, copy_end1;
  clock_t gpu_start, gpu_end;
  clock_t copy_start2, copy_end2;

  //3-D grid is 4x4x4, each block is 2x2x2
  dim3 block(1024, 1, 1);
  dim3 grid((len/block.x)+1, 1, 1);

  size_t byte_size = len * sizeof(int);
  h_arr1 = (int*)malloc(byte_size);
  h_arr2 = (int*)malloc(byte_size);
  h_gpu_result = (int*)malloc(byte_size);
  h_cpu_result = (int*)malloc(byte_size);
  for (i=0; i<len; i++) h_arr1[i]= i*2;
  for (i=0; i<len; i++) h_arr2[i]= i;
  
  gpuErrchk(cudaMalloc((void**)(&d_arr1), byte_size));
  gpuErrchk(cudaMalloc((void**)(&d_arr2), byte_size));
  gpuErrchk(cudaMalloc((void**)(&d_gpu_result), byte_size));

  copy_start1 = clock();
  gpuErrchk(cudaMemcpy(d_arr1, h_arr1, byte_size, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_arr2, h_arr2, byte_size, cudaMemcpyHostToDevice));
  copy_end1 = clock();

  gpu_start = clock();
  sum_array_gpu_1d << <grid, block>> > (d_arr1, d_arr2, d_gpu_result, len);
  //Wait until kernel call is completed
  cudaDeviceSynchronize();
  gpu_end = clock();

  copy_start2 = clock();
  cudaMemcpy(h_gpu_result, d_gpu_result, byte_size, cudaMemcpyDeviceToHost);
  copy_end2 = clock();

  //Also do it on cpu to verify results
  cpu_start = clock();
  sum_array_cpu_1d(h_arr1, h_arr2, h_cpu_result, len);
  cpu_end = clock();


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
  
  printf("CPU took %4.6f seconds.\n", ((double)(cpu_end-cpu_start))/CLOCKS_PER_SEC);
  printf("GPU took %4.6f seconds:\n", ((double)(copy_end1-copy_start1+gpu_end-gpu_start+copy_end2-copy_start2))/CLOCKS_PER_SEC);
  printf("  --> Host to device copying took %4.6f seconds.\n", ((double)(copy_end1-copy_start1))/CLOCKS_PER_SEC);
  printf("  --> Kernel took %4.6f seconds.\n", ((double)(gpu_end-gpu_start))/CLOCKS_PER_SEC);
  printf("  --> Device to host copying took %4.6f seconds.\n", ((double)(copy_end2-copy_start2))/CLOCKS_PER_SEC);

  return 0;
}