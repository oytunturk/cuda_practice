#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>

#include "my_tools.cuh"

//Add two arrays
//Also error handling
__global__ void sum_array_gpu_1d(int *a, int *b, int *c, int *result, size_t len) {
  int gid = blockDim.x * blockIdx.x + threadIdx.x;

  if (gid < len) {
    //printf("gid = %d\n", gid);
    result[gid] = a[gid] + b[gid] + c[gid];
  }
}

void sum_array_cpu_1d(int *a, int *b, int *c, int *result, int len) {
  for(int i=0; i<len; i++) {
    result[i] = a[i] + b[i] + c[i];
  }
}

void run_test(int *h_arr1, int *h_arr2, int *h_arr3, int len, int block_size, bool show_info, double* total_times) {
  int *d_arr1 = NULL;
  int *d_arr2 = NULL;
  int *d_arr3 = NULL;
  int *h_gpu_result = NULL;
  int *d_gpu_result = NULL;
  int *h_cpu_result = NULL;  
  int i; 
  clock_t cpu_start, cpu_end;
  clock_t copy_start1, copy_end1;
  clock_t gpu_start, gpu_end;
  clock_t copy_start2, copy_end2;

  //1-D grid and block
  dim3 block(block_size, 1, 1);
  dim3 grid((len/block.x)+1, 1, 1);

  size_t byte_size = len * sizeof(int);

  h_gpu_result = (int*)malloc(byte_size);
  h_cpu_result = (int*)malloc(byte_size);

  gpuErrchk(cudaMalloc((void**)(&d_arr1), byte_size));
  gpuErrchk(cudaMalloc((void**)(&d_arr2), byte_size));
  gpuErrchk(cudaMalloc((void**)(&d_arr3), byte_size));
  gpuErrchk(cudaMalloc((void**)(&d_gpu_result), byte_size));

  copy_start1 = clock();
  gpuErrchk(cudaMemcpy(d_arr1, h_arr1, byte_size, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_arr2, h_arr2, byte_size, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_arr3, h_arr3, byte_size, cudaMemcpyHostToDevice));
  copy_end1 = clock();

  gpu_start = clock();
  sum_array_gpu_1d << <grid, block>> > (d_arr1, d_arr2, d_arr3, d_gpu_result, len);
  //Wait until kernel call is completed
  cudaDeviceSynchronize();
  gpu_end = clock();

  copy_start2 = clock();
  cudaMemcpy(h_gpu_result, d_gpu_result, byte_size, cudaMemcpyDeviceToHost);
  copy_end2 = clock();

  //Also do it on cpu to verify results
  cpu_start = clock();
  sum_array_cpu_1d(h_arr1, h_arr2, h_arr3, h_cpu_result, len);
  cpu_end = clock();

  int err = 0;
  for(i=0; i<len; i++) {
    if (fabs(h_cpu_result[i]-h_gpu_result[i])>1.0e-10) {
      printf("ERROR! h_cpu_result[%d] = %d --> h_gpu_result[%d] = h_arr1[%d] + h_arr2[%d] = %d + %d = %d\n", i, h_cpu_result[i], i, i, i, h_arr1[i], h_arr2[i], h_gpu_result[i]);
    }
  }
  if (err==0 && show_info) {
    printf("Success! All gpu results match cpu results\n");
  }

  cudaFree((void*)d_arr1);
  cudaFree((void*)d_arr2);
  cudaFree((void*)d_arr3);
  cudaFree((void*)d_gpu_result);
  free(h_gpu_result);
  free(h_cpu_result);

  //Reset the device
  cudaDeviceReset();
  
  double cpu_total_time = ((double)(cpu_end-cpu_start))/CLOCKS_PER_SEC;
  double host2device_time = ((double)(copy_end1-copy_start1))/CLOCKS_PER_SEC;
  double gpu_kernel_time = ((double)(gpu_end-gpu_start))/CLOCKS_PER_SEC;
  double device2host_time = ((double)(copy_end2-copy_start2))/CLOCKS_PER_SEC;
  total_times[0] = cpu_total_time;
  total_times[1] = host2device_time + gpu_kernel_time + device2host_time;
  total_times[2] = gpu_kernel_time;
  if (show_info) {
    printf("CPU took %4.6f seconds.\n", total_times[0]);
    printf("GPU took %4.6f seconds:\n", total_times[1]);
    printf("  --> Host to device copying took %4.6f seconds.\n", host2device_time);
    printf("  --> Kernel took %4.6f seconds.\n", gpu_kernel_time);
    printf("  --> Device to host copying took %4.6f seconds.\n", device2host_time);
  }
}

int main() {
  int i;
  int block_sizes[] = {64, 128, 256, 512};
  double total_times[3];

  int len = 4194304;
  int *h_arr1 = NULL;
  int *h_arr2 = NULL;
  int *h_arr3 = NULL;

  time_t t;

  size_t byte_size = len * sizeof(int);
  h_arr1 = (int*)malloc(byte_size);
  h_arr2 = (int*)malloc(byte_size);
  h_arr3 = (int*)malloc(byte_size);

  srand((unsigned)time(&t));
  for (i=0; i<len; i++) h_arr1[i] = (int)(rand() & 0xFF);
  for (i=0; i<len; i++) h_arr2[i] = (int)(rand() & 0xFF);
  for (i=0; i<len; i++) h_arr3[i] = (int)(rand() & 0xFF);

  for (int i=0; i<4; i++) {
    run_test(h_arr1, h_arr2, h_arr3, len, block_sizes[i], false, total_times);
    printf("Block size %d, CPU=%4.6f sec, GPU=%4.6f sec, GPU_KERNEL=%4.6f sec\n", block_sizes[i], total_times[0], total_times[1], total_times[2]);
  }

  free(h_arr1);
  free(h_arr2);
  free(h_arr3);

  return 0;
}