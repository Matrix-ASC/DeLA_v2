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

template <uint64_t block_size, uint64_t atomic_size = 256>
__global__ void __launch_bounds__(block_size) la_spse_backward_kernel(
    float *f_grad,            //      B N C
    float *w_grad,            //      4 C
    const float *grad,        //      B N C
    const uint32_t *back_idx, //      B N C
    const float *input,       //      B N C
    const float *xyz,         //      B N 4
    const float *weight,      //      4 C
    const uint64_t N,
    const uint64_t C,
    const uint64_t BNC
){
    uint64_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= BNC) return;
    float proj[4];
    const uint64_t c = idx % C, 
                   bNn = idx / C,
                   b = bNn / N,
                   f_base = b*N*C + c;
    w_grad += bNn % atomic_size * 4 * C;
    // read projection weight
    for (uint64_t i=0; i<4; ++i)
    {
        proj[i] = weight[i*C + c];
    }
    // fix bias
    {
        const float4 center = *(float4*)(xyz + bNn*4);
        proj[0] += center.x;
        proj[1] += center.y;
        proj[2] += center.z;
    }
    // backward
    const uint32_t nbr = back_idx[idx];
    const float rel_f_ = (input[f_base + nbr*C] - input[idx]);
    const float g = grad[idx];
    const float4 nbr_xyz = *(float4*)(xyz + b*N*4 + nbr*4);
    const float x = proj[0] - nbr_xyz.x;
    const float y = proj[1] - nbr_xyz.y;
    const float z = proj[2] - nbr_xyz.z;
    const float dist = 1e-8f + x*x + y*y + z*z;

    const float edist = __expf(-(proj[3]*dist));

    // const float edist = __fdividef(1.0f, 1.0f + proj[3]*dist);

    // const float edist = rsqrtf(1.0f + proj[3]*dist);

    // const float edist = __expf(-(proj[3]*dist)) * (1.0f+proj[3]*dist);

    atomicAdd(&f_grad[f_base + nbr*C], g*edist);
    atomicAdd(&f_grad[idx], -(g*edist));

    const float rel_f = -(g * rel_f_ * edist);

    // const float rel_f = -(g * rel_f_ * edist * edist);

    // const float rel_f = -0.5f * (g * rel_f_ * edist * edist * edist);

    // const float rel_f = (-edist + __expf(-(proj[3]*dist))) * g * rel_f_;
    
    atomicAdd(&w_grad[3*C + c], rel_f * dist);
    const float rg = rel_f * proj[3] * 2.0f;
    atomicAdd(&w_grad[0*C + c], rg * x);
    atomicAdd(&w_grad[1*C + c], rg * y);
    atomicAdd(&w_grad[2*C + c], rg * z);
}


void la_spse_backward(
    torch::Tensor &f_grad,
    torch::Tensor &w_grad,
    const torch::Tensor &grad,
    const torch::Tensor &back_idx,
    const torch::Tensor &input,
    const torch::Tensor &xyz,
    const torch::Tensor &weight
){
    const uint64_t B = grad.size(0),
                   N = grad.size(1),
                   C = grad.size(2);
    constexpr uint64_t block_size = 256;
    la_spse_backward_kernel<block_size>
        <<<(B*N*C+block_size-1)/block_size, block_size>>>(
        (float*)f_grad.data_ptr(),
        (float*)w_grad.data_ptr(),
        (const float*)grad.data_ptr(),
        (const uint32_t*)back_idx.data_ptr(),
        (const float*)input.data_ptr(),
        (const float*)xyz.data_ptr(),
        (const float*)weight.data_ptr(),
        N, C, B*N*C
    );
    checkCudaError();
}