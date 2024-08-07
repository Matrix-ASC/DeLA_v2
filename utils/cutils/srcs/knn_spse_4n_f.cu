#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <assert.h>
#include <torch/extension.h>
#include "cuda_util.h"

/*
    threads in shared size process features within one point
*/

constexpr float eps = 1e-4 + 1.0;

__device__ static float calc_distance(
    const float4 xyz,
    const float proj[19]
){
    const float x = proj[9] + xyz.x * proj[0] + xyz.y * proj[3] + xyz.z * proj[6];
    const float y = proj[10] + xyz.x * proj[1] + xyz.y * proj[4] + xyz.z * proj[7];
    const float z = proj[11] + xyz.x * proj[2] + xyz.y * proj[5] + xyz.z * proj[8];
    return eps + x*x + y*y + z*z;
}

template <uint64_t shared_size, uint64_t block_size>
__global__ void __launch_bounds__(block_size) knn_spse_4n_forward_kernel(
    float *output,            //      B N C
    uint32_t *back_idx,       //      B N C
    const float *xyz,         //      B N 12
    const uint32_t *nbr_idx,  //      B N k
    const float *weight,      //      19 C
    const uint64_t N,
    const uint64_t C,
    const uint64_t k,
    const uint64_t BNC
){
    uint64_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= BNC) return;
    float proj[19];
    extern __shared__ float nbr_xyz[];  //  block_size / shared_size, k, 12
    const uint64_t c = idx % C, 
                   bNn = idx / C,
                   b = bNn / N;
    // read projection weight
    for (uint64_t i=0; i<19; ++i)
    {
        proj[i] = weight[i*C + c];
    }
    // fix bias
    {
        const float4 center = *(float4*)(xyz + bNn*12);
        const float x = center.x + proj[9],
                    y = center.y + proj[10],
                    z = center.z + proj[11];
        proj[9] = -(x*proj[0] + y*proj[3] + z*proj[6]);
        proj[10] = -(x*proj[1] + y*proj[4] + z*proj[7]);
        proj[11] = -(x*proj[2] + y*proj[5] + z*proj[8]);
    }
    // read nbr xyz
    float *nbr = nbr_xyz + threadIdx.x / shared_size * k * 12;
    for (uint64_t i=idx % shared_size; i<k*12; i+=shared_size)
    {
        const uint64_t nbr_i = nbr_idx[bNn*k + i/12];
        nbr[i] = xyz[b*N*12 + nbr_i*12 + i%12];
    }
    __syncwarp();
    // find min
    float min_dist = 1e8;
    uint64_t min_idx;
    for (uint64_t i=0; i<k; ++i)
    {
        float dist = calc_distance(*(float4*)(nbr + i*12), proj);
        const float4 nf = *(float4*)(nbr + i*12 + 4);
        const float4 norm = *(float4*)(nbr + i*12 + 8);
        const float f1 = nf.x - proj[12], f2 = nf.y - proj[13], f3 = nf.z - proj[14], f4 = nf.w - proj[15];
        dist = dist + f1*f1 + f2*f2 + f3*f3 + f4*f4 + norm.x*proj[16] + norm.y*proj[17] + norm.z*proj[18];
        if (min_dist > dist)
        {
            min_dist = dist;
            min_idx = i;
        }
    }
    output[idx] = min_dist;
    back_idx[idx] = nbr_idx[bNn*k + min_idx];
}

void knn_spse_4n_forward(
    torch::Tensor &output,
    torch::Tensor &back_idx,
    const torch::Tensor &xyz,
    const torch::Tensor &nbr_idx,
    const torch::Tensor &weight
){
    const uint64_t B = output.size(0),
                   N = output.size(1),
                   C = output.size(2),
                   k = nbr_idx.size(2);
    if (C % 32 == 0)
    {
        constexpr uint64_t block_size = 256;
        constexpr uint64_t shared_size = 32;
        knn_spse_4n_forward_kernel<shared_size, block_size>
            <<<(B*N*C+block_size-1)/block_size, block_size, block_size/shared_size*k*12*sizeof(float)>>>(
            (float*)output.data_ptr(),
            (uint32_t*)back_idx.data_ptr(),
            (const float*)xyz.data_ptr(),
            (const uint32_t*)nbr_idx.data_ptr(),
            (const float*)weight.data_ptr(),
            N, C, k, B*N*C
        );
    } else {
        constexpr uint64_t block_size = 256;
        constexpr uint64_t shared_size = 16;
        knn_spse_4n_forward_kernel<shared_size, block_size>
            <<<(B*N*C+block_size-1)/block_size, block_size, block_size/shared_size*k*12*sizeof(float)>>>(
            (float*)output.data_ptr(),
            (uint32_t*)back_idx.data_ptr(),
            (const float*)xyz.data_ptr(),
            (const uint32_t*)nbr_idx.data_ptr(),
            (const float*)weight.data_ptr(),
            N, C, k, B*N*C
        );
    }
    checkCudaError();
}