#ifndef __RMS_NORM_GEMM_SIMPLE_FUSED_CUH__
#define __RMS_NORM_GEMM_SIMPLE_FUSED_CUH__

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// 简化的融合kernel - 确保正确性优先
template<typename T>
__global__ void rms_norm_gemm_simple_fused_kernel(
    T* __restrict__ output,          // [M, N]
    const T* __restrict__ input,     // [M, K]
    const T* __restrict__ weight_matrix,  // [K, N] 
    const T* __restrict__ rms_weight,     // [K]
    const T* __restrict__ bias,           // [N]
    const int M, const int N, const int K,
    const float epsilon,
    const bool has_bias,
    const ptrdiff_t stride_a,
    const ptrdiff_t ldc,
    const ptrdiff_t ldb  // B是列主序，所以只需要一个stride
) {
    // 每个block处理一行输入
    const int row = blockIdx.x;
    if (row >= M) return;
    
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    
    // 输入行指针
    const T* input_row = input + row * stride_a;
    
    // Step 1: 计算RMS scale - 所有线程协作
    float sum_sq = 0.0f;
    for (int i = tid; i < K; i += num_threads) {
        float val = static_cast<float>(input_row[i]);
        sum_sq += val * val;
    }
    
    // Block内规约
    __shared__ float shared_sum[256];
    shared_sum[tid] = sum_sq;
    __syncthreads();
    
    // 规约求和
    for (int s = num_threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    // 计算RMS scale
    __shared__ float rms_scale;
    if (tid == 0) {
        rms_scale = rsqrtf(shared_sum[0] / K + epsilon);
    }
    __syncthreads();
    
    // Step 2: 应用RMSNorm到shared memory
    extern __shared__ char shared_mem_simple[];
    float* normed_row = reinterpret_cast<float*>(shared_mem_simple);
    
    for (int i = tid; i < K; i += num_threads) {
        float val = static_cast<float>(input_row[i]);
        float weight = static_cast<float>(rms_weight[i]);
        normed_row[i] = val * weight * rms_scale;
    }
    __syncthreads();
    
    // Step 3: 计算GEMM - 每个线程负责输出的一部分列
    for (int col = tid; col < N; col += num_threads) {
        float sum = 0.0f;
        
        // 计算点积 normed_row[K] · weight_matrix[:, col]
        for (int k = 0; k < K; k++) {
            // B矩阵是列主序：B[k, col] = weight_matrix[col * K + k]
            float b_val = static_cast<float>(weight_matrix[col * ldb + k]);
            sum += normed_row[k] * b_val;
        }
        
        // 添加bias
        if (has_bias && bias != nullptr) {
            sum += static_cast<float>(bias[col]);
        }
        
        // 写入输出
        output[row * ldc + col] = static_cast<T>(sum);
    }
}

// 优化版本 - 使用寄存器缓存和向量化
template<typename T>
__global__ void rms_norm_gemm_optimized_fused_kernel(
    T* __restrict__ output,
    const T* __restrict__ input,
    const T* __restrict__ weight_matrix,
    const T* __restrict__ rms_weight, 
    const T* __restrict__ bias,
    const int M, const int N, const int K,
    const float epsilon,
    const bool has_bias,
    const ptrdiff_t stride_a,
    const ptrdiff_t ldc,
    const ptrdiff_t ldb
) {
    // 使用2D grid - 每个block处理一个tile
    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;
    
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    
    // Tile大小
    const int TILE_M = 4;
    const int TILE_N = 64;
    
    const int global_row = block_row * TILE_M;
    const int global_col = block_col * TILE_N;
    
    if (global_row >= M || global_col >= N) return;
    
    // Shared memory分配
    extern __shared__ char shared_mem_opt[];
    float* rms_scales = reinterpret_cast<float*>(shared_mem_opt);
    float* normed_tile = rms_scales + TILE_M;
    
    // Step 1: 计算每行的RMS scale
    for (int row_in_tile = 0; row_in_tile < TILE_M && global_row + row_in_tile < M; row_in_tile++) {
        const int row = global_row + row_in_tile;
        const T* input_row = input + row * stride_a;
        
        float sum_sq = 0.0f;
        for (int i = tid; i < K; i += num_threads) {
            float val = static_cast<float>(input_row[i]);
            sum_sq += val * val;
        }
        
        // Warp内规约
        for (int offset = 16; offset > 0; offset /= 2) {
            sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
        }
        
        if (tid % 32 == 0) {
            atomicAdd(&rms_scales[row_in_tile], sum_sq);
        }
    }
    __syncthreads();
    
    // 完成RMS scale计算
    if (tid < TILE_M) {
        int row_in_tile = tid;
        if (global_row + row_in_tile < M) {
            rms_scales[row_in_tile] = rsqrtf(rms_scales[row_in_tile] / K + epsilon);
        }
    }
    __syncthreads();
    
    // Step 2: 计算GEMM tiles
    for (int row_in_tile = 0; row_in_tile < TILE_M && global_row + row_in_tile < M; row_in_tile++) {
        const int row = global_row + row_in_tile;
        const T* input_row = input + row * stride_a;
        const float scale = rms_scales[row_in_tile];
        
        // 每个线程处理多个输出列
        const int cols_per_thread = (TILE_N + num_threads - 1) / num_threads;
        
        for (int c = 0; c < cols_per_thread; c++) {
            int col = global_col + tid + c * num_threads;
            if (col < N) {
                float sum = 0.0f;
                
                // 向量化计算点积
                const int K_VEC = 4;
                const int k_vec = K / K_VEC * K_VEC;
                
                for (int k = 0; k < k_vec; k += K_VEC) {
                    float4 a_vals, w_vals, b_vals;
                    
                    // 加载并应用RMSNorm
                    #pragma unroll
                    for (int v = 0; v < K_VEC; v++) {
                        float inp = static_cast<float>(input_row[k + v]);
                        float wgt = static_cast<float>(rms_weight[k + v]);
                        float normed = inp * wgt * scale;
                        
                        // 加载B矩阵值（列主序）
                        float b_val = static_cast<float>(weight_matrix[col * ldb + k + v]);
                        
                        sum += normed * b_val;
                    }
                }
                
                // 处理剩余元素
                for (int k = k_vec; k < K; k++) {
                    float inp = static_cast<float>(input_row[k]);
                    float wgt = static_cast<float>(rms_weight[k]);
                    float normed = inp * wgt * scale;
                    float b_val = static_cast<float>(weight_matrix[col * ldb + k]);
                    sum += normed * b_val;
                }
                
                // 添加bias
                if (has_bias && bias != nullptr) {
                    sum += static_cast<float>(bias[col]);
                }
                
                // 写入输出
                output[row * ldc + col] = static_cast<T>(sum);
            }
        }
    }
}

#endif // __RMS_NORM_GEMM_SIMPLE_FUSED_CUH__