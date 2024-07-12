#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <assert.h>
#include <torch/extension.h>
#include "cuda_util.h"

/*
    1. shared allo in func
    2. read nbr idx in shared
    3. 3/8 f share SP
*/

/*
    threads in shared size process features within one point
*/

constexpr uint64_t group_size = 4;

struct __builtin_align__(group_size*4) floatn
{
    float v[group_size];
};

__device__ static float calc_distance(
    const float4 xyz,
    const float proj[4]
){
    const float x = xyz.x - proj[0];
    const float y = xyz.y - proj[1];
    const float z = xyz.z - proj[2];
    return 1e-8f + x*x + y*y + z*z;
}

template <uint64_t block_size>
__global__ void __launch_bounds__(block_size) la_spse_a4_forward_kernel(
    float *output,            //      B N C
    uint32_t *back_idx,       //      B N C
    const float *input,       //      B N C
    const float *xyz,         //      B N 4
    const uint32_t *nbr_idx,  //      B N k
    const float *weight,      //      4 C/4
    const uint64_t N,
    const uint64_t C_,        //      C = 4 C_
    const uint64_t k,
    const uint64_t BNC_
){
    uint64_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= BNC_) return;
    float proj[4];
    const uint64_t c_ = idx % C_, 
                   c = c_ * 4,
                   C = C_ * 4,
                   bNn = idx / C_,
                   n = bNn % N,
                   b = bNn / N,
                   f_base = b*N*C + c,
                   n_base = bNn*k;
    // read projection weight
    for (uint64_t i=0; i<4; ++i)
    {
        proj[i] = weight[i*C_ + c_];
    }
    // fix bias
    {
        const float4 center = *(float4*)(xyz + bNn*4);
        proj[0] += center.x;
        proj[1] += center.y;
        proj[2] += center.z;
    }
    // pool
    const floatn cf = *(floatn*)(input + f_base + n*C);
    floatn max_val{-1e8, -1e8, -1e8, -1e8};
    uint32_t max_idx[group_size];
    for (uint64_t i=0; i<k; ++i)
    {
        const uint64_t nidx = nbr_idx[n_base+i];
        const floatn valn = *(floatn*)(input + f_base + nidx*C);
        const float dist = calc_distance(*(float4*)(xyz + b*N*4 + nidx*4), proj) * proj[3];
        // const float mul = __expf(-dist) * (1.0f+dist);
        const float mul = __expf(-dist);
        // const float mul = __fdividef(1.0f, 1.0f + dist);
        // const float mul = rsqrtf(1.0f + dist);
        for (uint64_t f_idx = 0; f_idx < group_size; ++f_idx)
        {
            const float val = (valn.v[f_idx] - cf.v[f_idx]) * mul;
            if (val > max_val.v[f_idx])
            {
                max_val.v[f_idx] = val;
                max_idx[f_idx] = nidx;
            }
        }
    }
    for (uint64_t f_idx = 0; f_idx < group_size; ++f_idx)
    {
        back_idx[f_base + n * C + f_idx] = max_idx[f_idx];
    }
    *(floatn*)(output + f_base + n * C) = max_val;
}


void la_spse_a4_forward(
    torch::Tensor &output,
    torch::Tensor &back_idx,
    const torch::Tensor &input,
    const torch::Tensor &xyz,
    const torch::Tensor &nbr_idx,
    const torch::Tensor &weight
){
    const uint64_t B = output.size(0),
                   N = output.size(1),
                   C = output.size(2) / 4,
                   k = nbr_idx.size(2);

    constexpr uint64_t block_size = 512;
    la_spse_a4_forward_kernel<block_size>
        <<<(B*N*C+block_size-1)/block_size, block_size>>>(
        (float*)output.data_ptr(),
        (uint32_t*)back_idx.data_ptr(),
        (const float*)input.data_ptr(),
        (const float*)xyz.data_ptr(),
        (const uint32_t*)nbr_idx.data_ptr(),
        (const float*)weight.data_ptr(),
        N, C, k, B*N*C
    );
    
    checkCudaError();
}