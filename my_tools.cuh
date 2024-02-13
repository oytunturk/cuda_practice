#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

//Helper function for error checking quickly
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t err_code, const char *file, int line, bool abort = true) {
  if (err_code != cudaSuccess) {
    fprintf(stderr, "Error! In file %s at line %d --> %s\n", file, line, cudaGetErrorString(err_code));
    if(abort) exit(err_code);
  }
}

void compare_arrays(float * a, float * b, float size)
{
	for (int i = 0; i < size; i++)
	{
		if (a[i] != b[i])
		{
			printf("Arrays are different \n");
			
			return;
		}
	}
	printf("Arrays are same \n");
}

void sum_array_cpu(float* a, float* b, float *c, int size)
{
	for (int i = 0; i < size; i++)
	{
		c[i] = a[i] + b[i];
	}
}