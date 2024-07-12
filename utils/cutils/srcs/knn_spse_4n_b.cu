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

constexpr float eps = 1e-8;

template <uint64_t block_size, uint64_t atomic_size = 256>
__global__ void __launch_bounds__(block_size) knn_spse_4n_backward_kernel(
    float *wgrad,             //      A 19 C
    const float *grad,        //      B N C
    const uint32_t *back_idx, //      B N C
    const float *kxyz,        //      B N 12
    const float *weight,      //      19 C
    const uint64_t N,
    const uint64_t C,
    const uint64_t BNC
){
    uint64_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= BNC) return;
    float proj[19];
    const uint64_t c = idx % C, 
                   bNn = idx / C,
                   b = bNn / N;
    wgrad += bNn % atomic_size * 19 * C;
    // read projection weight
    for (uint64_t i=0; i<19; ++i)
    {
        proj[i] = weight[i*C + c];
    }
    // fix bias
    const float4 center = *(float4*)(kxyz + bNn*12);
    const float p9 = proj[9], p10 = proj[10], p11 = proj[11];
    {
        const float x = center.x + proj[9],
                    y = center.y + proj[10],
                    z = center.z + proj[11];
        proj[9] = -(x*proj[0] + y*proj[3] + z*proj[6]);
        proj[10] = -(x*proj[1] + y*proj[4] + z*proj[7]);
        proj[11] = -(x*proj[2] + y*proj[5] + z*proj[8]);
    }
    // backward min_dist
    float proj_grad[12];
    const float min_dist_grad = grad[idx] * 2.0f;
    {
        const float4 nf = *(float4*)(kxyz + b*N*12 + back_idx[idx]*12 + 4);
        const float4 norm = *(float4*)(kxyz + b*N*12 + back_idx[idx]*12 + 8);
        const float f1 = proj[12] - nf.x, f2 = proj[13] - nf.y, f3 = proj[14] - nf.z, f4 = proj[15] - nf.w;
        atomicAdd(wgrad + 12*C + c, min_dist_grad * f1);
        atomicAdd(wgrad + 13*C + c, min_dist_grad * f2);
        atomicAdd(wgrad + 14*C + c, min_dist_grad * f3);
        atomicAdd(wgrad + 15*C + c, min_dist_grad * f4);
        const float ngrad = grad[idx];
        atomicAdd(wgrad + 16*C + c, ngrad * norm.x);
        atomicAdd(wgrad + 17*C + c, ngrad * norm.y);
        atomicAdd(wgrad + 18*C + c, ngrad * norm.z);
    }
    {
        const float4 xyz = *(float4*)(kxyz + b*N*12 + back_idx[idx]*12);
        const float x = proj[9] + xyz.x * proj[0] + xyz.y * proj[3] + xyz.z * proj[6],
                    y = proj[10] + xyz.x * proj[1] + xyz.y * proj[4] + xyz.z * proj[7],
                    z = proj[11] + xyz.x * proj[2] + xyz.y * proj[5] + xyz.z * proj[8];
        const float bx = min_dist_grad * x,
                    by = min_dist_grad * y,
                    bz = min_dist_grad * z;
        proj_grad[9] = bx;  proj_grad[10] = by;  proj_grad[11] = bz;
        proj_grad[0] = bx * xyz.x;  proj_grad[3] = bx * xyz.y;  proj_grad[6] = bx * xyz.z;
        proj_grad[1] = by * xyz.x;  proj_grad[4] = by * xyz.y;  proj_grad[7] = by * xyz.z;
        proj_grad[2] = bz * xyz.x;  proj_grad[5] = bz * xyz.y;  proj_grad[8] = bz * xyz.z;
    }
    // backward proj
    // backward fix bias
    {
        const float p9g = proj_grad[9] * proj[0] + proj_grad[10] * proj[1] + proj_grad[11] * proj[2];
        atomicAdd(wgrad + 9*C + c, -p9g);
        const float p10g = proj_grad[9] * proj[3] + proj_grad[10] * proj[4] + proj_grad[11] * proj[5];
        atomicAdd(wgrad + 10*C + c, -p10g);
        const float p11g = proj_grad[9] * proj[6] + proj_grad[10] * proj[7] + proj_grad[11] * proj[8];
        atomicAdd(wgrad + 11*C + c, -p11g);
        const float x = -(center.x + p9),
                    y = -(center.y + p10),
                    z = -(center.z + p11);
        proj_grad[0] += proj_grad[9] * x;
        proj_grad[1] += proj_grad[10] * x;
        proj_grad[2] += proj_grad[11] * x;
        proj_grad[3] += proj_grad[9] * y;
        proj_grad[4] += proj_grad[10] * y;
        proj_grad[5] += proj_grad[11] * y;
        proj_grad[6] += proj_grad[9] * z;
        proj_grad[7] += proj_grad[10] * z;
        proj_grad[8] += proj_grad[11] * z;
    }
    for (uint64_t i=0; i<9; ++i){
        atomicAdd(wgrad + i*C + c, proj_grad[i]);
    }
}

void knn_spse_4n_backward(
    torch::Tensor &wgrad,
    const torch::Tensor &grad,
    const torch::Tensor &back_idx,
    const torch::Tensor &xyz,
    const torch::Tensor &weight
){
    const uint64_t B = grad.size(0),
                   N = grad.size(1),
                   C = grad.size(2);
    constexpr uint64_t block_size = 256;
    knn_spse_4n_backward_kernel<block_size>
        <<<(B*N*C+block_size-1)/block_size, block_size>>>(
        (float*)wgrad.data_ptr(),
        (const float*)grad.data_ptr(),
        (const uint32_t*)back_idx.data_ptr(),
        (const float*)xyz.data_ptr(),
        (const float*)weight.data_ptr(),
        N, C, B*N*C
    );
    checkCudaError();
}