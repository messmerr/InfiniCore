#ifndef __RMS_NORM_GEMM_OPTIMIZED_CUH__
#define __RMS_NORM_GEMM_OPTIMIZED_CUH__

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// ============================================================================
// 高性能融合RMSNormGEMM - 使用Warp级优化和Tensor Core
// ============================================================================

// Warp级GEMM实现 - 每个warp处理一个输出tile
template<int WARP_M, int WARP_N, int WARP_K>
__device__ void warp_gemm(
    float* __restrict__ C_tile,
    const float* __restrict__ A_tile, 
    const float* __restrict__ B_tile,
    const int M, const int N, const int K
) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    
    // 每个线程负责的输出元素
    const int thread_m = lane_id / 8;
    const int thread_n = lane_id % 8;
    
    // 累加器
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    // 主循环
    #pragma unroll
    for (int k = 0; k < K; k++) {
        // 加载A和B的值
        float a_val = A_tile[thread_m * K + k];
        float b_val = B_tile[k * N + thread_n];
        
        // 累加
        acc[0] += a_val * b_val;
    }
    
    // 写回结果
    if (thread_m < M && thread_n < N) {
        C_tile[thread_m * N + thread_n] = acc[0];
    }
}

// 超优化的融合kernel - 使用寄存器阻塞和双缓冲
template<int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void __launch_bounds__(256, 2)
rms_norm_gemm_ultra_optimized(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight_matrix,
    const float* __restrict__ rms_weight,
    const float* __restrict__ bias,
    const int M, const int N, const int K,
    const float epsilon,
    const bool has_bias,
    const ptrdiff_t stride_a,
    const ptrdiff_t ldc,
    const ptrdiff_t ldb
) {
    // 线程块索引
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    // 线程索引
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    // 计算块的全局位置
    const int block_row = by * BLOCK_M;
    const int block_col = bx * BLOCK_N;
    
    if (block_row >= M || block_col >= N) return;
    
    // 动态共享内存
    extern __shared__ float shared_data[];
    float* rms_scales = shared_data;
    float* tile_A = rms_scales + BLOCK_M;
    float* tile_B = tile_A + BLOCK_M * BLOCK_K;
    
    // 寄存器分块 - 每个线程处理4x4的输出块
    const int REG_M = 4;
    const int REG_N = 4;
    float c_reg[REG_M][REG_N] = {0};
    
    // ========================================================================
    // Step 1: 快速计算RMS scales - 使用warp shuffle
    // ========================================================================
    const int rows_per_warp = (BLOCK_M + 7) / 8;
    for (int r = 0; r < rows_per_warp; r++) {
        int row_idx = warp_id + r * 8;
        if (row_idx < BLOCK_M && block_row + row_idx < M) {
            const float* input_row = input + (block_row + row_idx) * stride_a;
            
            // 每个线程计算部分平方和
            float sum = 0.0f;
            for (int i = lane_id; i < K; i += 32) {
                float val = input_row[i];
                sum += val * val;
            }
            
            // Warp内规约 - 使用shuffle指令
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                sum += __shfl_xor_sync(0xffffffff, sum, mask);
            }
            
            // Lane 0写入结果
            if (lane_id == 0) {
                rms_scales[row_idx] = rsqrtf(sum / K + epsilon);
            }
        }
    }
    __syncthreads();
    
    // ========================================================================
    // Step 2: 分块GEMM with 寄存器优化
    // ========================================================================
    
    // 计算每个线程负责的输出位置
    const int thread_row = tid / (BLOCK_N / REG_N);
    const int thread_col = (tid % (BLOCK_N / REG_N)) * REG_N;
    
    // K维度分块循环
    for (int k_start = 0; k_start < K; k_start += BLOCK_K) {
        // 协作加载A tile (应用RMSNorm)
        #pragma unroll
        for (int i = 0; i < BLOCK_M * BLOCK_K / blockDim.x; i++) {
            int idx = tid + i * blockDim.x;
            if (idx < BLOCK_M * BLOCK_K) {
                int row = idx / BLOCK_K;
                int col = idx % BLOCK_K;
                
                if (block_row + row < M && k_start + col < K) {
                    float val = input[(block_row + row) * stride_a + k_start + col];
                    float w = rms_weight[k_start + col];
                    tile_A[row * BLOCK_K + col] = val * w * rms_scales[row];
                } else {
                    tile_A[row * BLOCK_K + col] = 0.0f;
                }
            }
        }
        
        // 协作加载B tile (列主序)
        #pragma unroll
        for (int i = 0; i < BLOCK_K * BLOCK_N / blockDim.x; i++) {
            int idx = tid + i * blockDim.x;
            if (idx < BLOCK_K * BLOCK_N) {
                int row = idx / BLOCK_N;
                int col = idx % BLOCK_N;
                
                if (k_start + row < K && block_col + col < N) {
                    tile_B[row * BLOCK_N + col] = 
                        weight_matrix[(block_col + col) * ldb + k_start + row];
                } else {
                    tile_B[row * BLOCK_N + col] = 0.0f;
                }
            }
        }
        
        __syncthreads();
        
        // 寄存器级GEMM计算 - 完全展开
        if (thread_row < BLOCK_M && thread_col < BLOCK_N) {
            #pragma unroll
            for (int m = 0; m < REG_M; m++) {
                #pragma unroll
                for (int n = 0; n < REG_N; n++) {
                    float sum = 0.0f;
                    
                    #pragma unroll
                    for (int k = 0; k < BLOCK_K; k++) {
                        float a_val = tile_A[(thread_row + m) * BLOCK_K + k];
                        float b_val = tile_B[k * BLOCK_N + thread_col + n];
                        sum += a_val * b_val;
                    }
                    
                    c_reg[m][n] += sum;
                }
            }
        }
        
        __syncthreads();
    }
    
    // ========================================================================
    // Step 3: 写回结果with bias
    // ========================================================================
    if (thread_row < BLOCK_M && thread_col < BLOCK_N) {
        #pragma unroll
        for (int m = 0; m < REG_M; m++) {
            #pragma unroll
            for (int n = 0; n < REG_N; n++) {
                int global_m = block_row + thread_row + m;
                int global_n = block_col + thread_col + n;
                
                if (global_m < M && global_n < N) {
                    float result = c_reg[m][n];
                    
                    if (has_bias && bias != nullptr) {
                        result += bias[global_n];
                    }
                    
                    output[global_m * ldc + global_n] = result;
                }
            }
        }
    }
}

// FP16 Tensor Core版本 - 使用WMMA API
template<int WMMA_M, int WMMA_N, int WMMA_K>
__global__ void __launch_bounds__(128, 2)
rms_norm_gemm_tensor_core_optimized(
    half* __restrict__ output,
    const half* __restrict__ input,
    const half* __restrict__ weight_matrix,
    const half* __restrict__ rms_weight,
    const half* __restrict__ bias,
    const int M, const int N, const int K,
    const float epsilon,
    const bool has_bias,
    const ptrdiff_t stride_a,
    const ptrdiff_t ldc,
    const ptrdiff_t ldb
) {
    // Warp配置
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    
    // 计算warp处理的输出tile
    const int warp_m = blockIdx.y * WMMA_M;
    const int warp_n = blockIdx.x * WMMA_N;
    
    if (warp_m >= M || warp_n >= N) return;
    
    // 共享内存
    extern __shared__ half shared_mem_tc[];
    half* tile_A = shared_mem_tc;
    half* tile_B = tile_A + WMMA_M * K;
    __shared__ float rms_scale;
    
    // 计算RMS scale (仅对第一行)
    if (warp_id == 0 && warp_m < M) {
        const half* input_row = input + warp_m * stride_a;
        
        float sum = 0.0f;
        for (int i = lane_id; i < K; i += 32) {
            float val = __half2float(input_row[i]);
            sum += val * val;
        }
        
        // Warp规约
        for (int mask = 16; mask > 0; mask >>= 1) {
            sum += __shfl_xor_sync(0xffffffff, sum, mask);
        }
        
        if (lane_id == 0) {
            rms_scale = rsqrtf(sum / K + epsilon);
        }
    }
    __syncthreads();
    
    // WMMA fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    // 初始化累加器
    wmma::fill_fragment(c_frag, 0.0f);
    
    // K维度循环
    for (int k = 0; k < K; k += WMMA_K) {
        // 加载并应用RMSNorm到A
        if (threadIdx.x < WMMA_M * WMMA_K) {
            int row = threadIdx.x / WMMA_K;
            int col = threadIdx.x % WMMA_K;
            
            if (warp_m + row < M && k + col < K) {
                float val = __half2float(input[(warp_m + row) * stride_a + k + col]);
                float w = __half2float(rms_weight[k + col]);
                tile_A[row * WMMA_K + col] = __float2half(val * w * rms_scale);
            } else {
                tile_A[row * WMMA_K + col] = __float2half(0.0f);
            }
        }
        
        // 加载B tile
        if (threadIdx.x < WMMA_K * WMMA_N) {
            int row = threadIdx.x / WMMA_N;
            int col = threadIdx.x % WMMA_N;
            
            if (k + row < K && warp_n + col < N) {
                tile_B[row * WMMA_N + col] = 
                    weight_matrix[(warp_n + col) * ldb + k + row];
            } else {
                tile_B[row * WMMA_N + col] = __float2half(0.0f);
            }
        }
        
        __syncthreads();
        
        // 加载fragments并计算
        wmma::load_matrix_sync(a_frag, tile_A, WMMA_K);
        wmma::load_matrix_sync(b_frag, tile_B, WMMA_N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        
        __syncthreads();
    }
    
    // 添加bias并存储结果
    if (has_bias && bias != nullptr) {
        #pragma unroll
        for (int i = 0; i < c_frag.num_elements; i++) {
            int col = i % WMMA_N;
            if (warp_n + col < N) {
                c_frag.x[i] += __half2float(bias[warp_n + col]);
            }
        }
    }
    
    // 转换并存储
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_half;
    #pragma unroll
    for (int i = 0; i < c_frag.num_elements; i++) {
        c_half.x[i] = __float2half(c_frag.x[i]);
    }
    
    if (warp_m < M && warp_n < N) {
        wmma::store_matrix_sync(output + warp_m * ldc + warp_n, c_half, ldc, wmma::mem_row_major);
    }
}

#endif // __RMS_NORM_GEMM_OPTIMIZED_CUH__