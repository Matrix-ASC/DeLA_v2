#ifndef CUDA_UTIL_H_
#define CUDA_UTIL_H_

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

static void checkCudaError(){
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}


#endif  //  CUDA_UTIL_H_