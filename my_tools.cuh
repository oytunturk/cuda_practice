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