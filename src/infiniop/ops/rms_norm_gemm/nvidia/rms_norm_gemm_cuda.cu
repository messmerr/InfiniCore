#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "rms_norm_gemm_cuda.cuh"
#include "../cuda/kernel.cuh"
#include "rms_norm_gemm_tensor_core.cuh"
#include "rms_norm_gemm_simple_fused.cuh"
#include "rms_norm_gemm_optimized.cuh"
#include <cublas_v2.h>
#include <type_traits>

// ============================================================================
// 高性能融合算子实现：两阶段策略 
// 阶段1: 优化的RMSNorm kernel
// 阶段2: 调用高性能cuBLAS GEMM
// ============================================================================

// RMSNorm kernel (F32 path) - 只做标准化，输出到workspace (float)
template <typename Tdata, typename Tweight, int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE, 2) void optimized_rms_norm_kernel_f32(
    float *__restrict__ normed_output,      // workspace中的标准化输出 (float)
    const Tdata *__restrict__ input,        // 输入数据
    const Tweight *__restrict__ weights,    // RMSNorm权重
    const size_t m, const size_t k,
    const float epsilon,
    const ptrdiff_t input_stride) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= (int)m) return;

    const Tdata *input_row = input + row * input_stride;
    float *output_row = normed_output + row * k;

    // sum of squares with vectorized loading
    float thread_sum = 0.0f;
    const int k_vec4 = (k / 4) * 4;
    
    // Vectorized accumulation for float4
    for (int i = tid * 4; i < k_vec4; i += BLOCK_SIZE * 4) {
        float4 vals = *reinterpret_cast<const float4*>(&input_row[i]);
        thread_sum += vals.x * vals.x;
        thread_sum += vals.y * vals.y;
        thread_sum += vals.z * vals.z;
        thread_sum += vals.w * vals.w;
    }
    
    // Handle remaining elements
    for (int i = k_vec4 + tid; i < (int)k; i += BLOCK_SIZE) {
        float val = static_cast<float>(input_row[i]);
        thread_sum += val * val;
    }

    __shared__ float sdata[BLOCK_SIZE];
    sdata[tid] = thread_sum;
    __syncthreads();

    // Warp-level reduction for better performance
    #pragma unroll
    for (int s = BLOCK_SIZE / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    
    // Final warp reduction without __syncthreads
    if (tid < 32) {
        volatile float *vdata = sdata;
        if (BLOCK_SIZE >= 64) vdata[tid] += vdata[tid + 32];
        if (BLOCK_SIZE >= 32) vdata[tid] += vdata[tid + 16];
        if (BLOCK_SIZE >= 16) vdata[tid] += vdata[tid + 8];
        if (BLOCK_SIZE >= 8) vdata[tid] += vdata[tid + 4];
        if (BLOCK_SIZE >= 4) vdata[tid] += vdata[tid + 2];
        if (BLOCK_SIZE >= 2) vdata[tid] += vdata[tid + 1];
    }

    __shared__ float shared_rms;
    if (tid == 0) shared_rms = rsqrtf(sdata[0] / (float)k + epsilon);
    __syncthreads();

    const float rms = shared_rms;
    
    // Aggressive vectorization with float4 for better memory throughput
    const int VEC_SIZE = 4;
    const int k_vec = (int)k / VEC_SIZE;
    
    // Process multiple float4 per thread for better occupancy
    const int VECS_PER_THREAD = 2;
    for (int base_i = tid * VECS_PER_THREAD; base_i < k_vec; base_i += BLOCK_SIZE * VECS_PER_THREAD) {
        #pragma unroll
        for (int v = 0; v < VECS_PER_THREAD && (base_i + v) < k_vec; v++) {
            const int i = base_i + v;
            float4 input_vec = reinterpret_cast<const float4*>(input_row)[i];
            float4 weight_vec = reinterpret_cast<const float4*>(weights)[i];
            float4 output_vec;
            output_vec.x = input_vec.x * weight_vec.x * rms;
            output_vec.y = input_vec.y * weight_vec.y * rms;
            output_vec.z = input_vec.z * weight_vec.z * rms;
            output_vec.w = input_vec.w * weight_vec.w * rms;
            reinterpret_cast<float4*>(output_row)[i] = output_vec;
        }
    }
    
    // Handle remaining elements
    for (int i = k_vec * VEC_SIZE + tid; i < (int)k; i += BLOCK_SIZE) {
        float input_val = static_cast<float>(input_row[i]);
        float weight_val = static_cast<float>(weights[i]);
        output_row[i] = input_val * weight_val * rms;
    }
}

// RMSNorm kernel (F16 path) - 输出到workspace (half)
template <typename WtT>
__global__ __launch_bounds__(512, 2) void optimized_rms_norm_kernel_f16(
    half *__restrict__ normed_output,       // workspace half
    const half *__restrict__ input,         // 输入数据 half
    const WtT *__restrict__ weights,        // 权重 half/float
    const size_t m, const size_t k,
    const float epsilon,
    const ptrdiff_t input_stride) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= (int)m) return;

    const half *input_row = input + row * input_stride;
    half *output_row = normed_output + row * k;

    // sum of squares in float with vectorized loading
    float thread_sum = 0.0f;
    
    // Use half2 for vectorized accumulation
    const int k2 = k / 2;
    for (int i = tid; i < k2; i += blockDim.x) {
        half2 h2 = reinterpret_cast<const half2*>(input_row)[i];
        float2 f2 = __half22float2(h2);
        thread_sum += f2.x * f2.x + f2.y * f2.y;
    }
    
    // Handle odd element
    if ((k & 1) && tid == 0) {
        float v = __half2float(input_row[k - 1]);
        thread_sum += v * v;
    }
    
    __shared__ float sdata[1024]; // upper bound, safe for <=1024 threads
    sdata[tid] = thread_sum;
    __syncthreads();
    
    // Warp-level reduction
    #pragma unroll
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    
    // Final warp reduction
    if (tid < 32) {
        volatile float *vdata = sdata;
        if (blockDim.x >= 64) vdata[tid] += vdata[tid + 32];
        if (blockDim.x >= 32) vdata[tid] += vdata[tid + 16];
        if (blockDim.x >= 16) vdata[tid] += vdata[tid + 8];
        if (blockDim.x >= 8) vdata[tid] += vdata[tid + 4];
        if (blockDim.x >= 4) vdata[tid] += vdata[tid + 2];
        if (blockDim.x >= 2) vdata[tid] += vdata[tid + 1];
    }
    __shared__ float shared_rms;
    if (tid == 0) shared_rms = rsqrtf(sdata[0] / (float)k + epsilon);
    __syncthreads();

    // Optimized half2 output with better memory coalescing
    const float rms = shared_rms;
    const int VECS_PER_THREAD = 2;
    
    for (int base_i = tid * VECS_PER_THREAD; base_i < k2; base_i += blockDim.x * VECS_PER_THREAD) {
        #pragma unroll
        for (int v = 0; v < VECS_PER_THREAD && (base_i + v) < k2; v++) {
            int i = base_i + v;
            // load 2 inputs
            half2 in2 = reinterpret_cast<const half2*>(input_row)[i];
            float2 f2 = __half22float2(in2);
            
            // load 2 weights (support WtT=half or float)
            float gw0, gw1;
            if constexpr (std::is_same<WtT, half>::value) {
                half2 w2 = reinterpret_cast<const half2*>(weights)[i];
                float2 wf2 = __half22float2(w2);
                gw0 = wf2.x;
                gw1 = wf2.y;
            } else {
                const float2 w2 = reinterpret_cast<const float2*>(weights)[i];
                gw0 = w2.x;
                gw1 = w2.y;
            }
            
            float o0 = f2.x * gw0 * rms;
            float o1 = f2.y * gw1 * rms;
            reinterpret_cast<half2*>(output_row)[i] = __floats2half2_rn(o0, o1);
        }
    }
    // tail element when k is odd
    if ((k & 1) != 0) {
        int last = (int)k - 1;
        if (tid == 0) {
            float gw;
            if constexpr (std::is_same<WtT, half>::value) {
                gw = __half2float(reinterpret_cast<const half*>(weights)[last]);
            } else {
                gw = reinterpret_cast<const float*>(weights)[last];
            }
            float o = __half2float(input_row[last]) * gw * shared_rms;
            output_row[last] = __float2half_rn(o);
        }
    }
}

// Bias addition kernel
template <typename Tdata>
__global__ void add_bias_kernel(
    Tdata *__restrict__ output,       // [m, n]  
    const Tdata *__restrict__ bias,   // [n]
    const size_t m, const size_t n,
    const ptrdiff_t ldc) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = m * n;
    
    if (idx < total_elements) {
        const int row = idx / n;
        const int col = idx % n;
        output[row * ldc + col] += bias[col];
    }
}

template <typename Tdata, typename Tweight, typename Tcompute, unsigned int BLOCK_SIZE>
INFINIOP_CUDA_KERNEL rms_norm_gemm_kernel(
    Tdata *__restrict__ c, 
    const Tdata *__restrict__ b, 
    const Tdata *__restrict__ a, 
    const Tweight *__restrict__ w,
    const Tdata *__restrict__ bias,
    const size_t m, const size_t n, const size_t k,
    float epsilon,
    const ptrdiff_t stride_a,   
    const ptrdiff_t ldc,
    const ptrdiff_t ldb_row, 
    const ptrdiff_t ldb_col,
    bool has_bias
) {
    rms_norm_gemm_block<Tdata, Tweight, Tcompute, BLOCK_SIZE> (
        c,b,a,w,bias,m,n,k,epsilon,stride_a,ldc,ldb_row,ldb_col,has_bias
    );
}

namespace op::rms_norm_gemm::cuda {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t w_desc,
    infiniopTensorDescriptor_t bias_desc,
    float epsilon) {
    auto result = RMSNormGemmInfo::create(c_desc, a_desc, b_desc, w_desc, bias_desc, epsilon);
    CHECK_RESULT(result);
    
    // 计算workspace大小：F32和F16均需要额外workspace
    auto info = result.take();
    size_t workspace_size = 0;
    // F32: 使用float workspace；F16: 使用half workspace
    if (info.rms_norm_info.atype == INFINI_DTYPE_F32 && info.rms_norm_info.wtype == INFINI_DTYPE_F32) {
        size_t m = info.gemm_info.m;
        size_t k = info.gemm_info.k;
        workspace_size = m * k * sizeof(float);
    } else if (info.rms_norm_info.atype == INFINI_DTYPE_F16 && (info.rms_norm_info.wtype == INFINI_DTYPE_F16 || info.rms_norm_info.wtype == INFINI_DTYPE_F32)) {
        size_t m = info.gemm_info.m;
        size_t k = info.gemm_info.k;
        workspace_size = m * k * sizeof(half);
    }
    
    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        std::move(info), workspace_size, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <unsigned int BLOCK_SIZE>
infiniStatus_t launchKernel(
        void *c,                
        const void *b,          
        const void *a,
        const void *w,
        const void *bias,
        const size_t m, const size_t n, const size_t k,
        float epsilon,
        const ptrdiff_t stride_a,   
        const ptrdiff_t ldc,
        const ptrdiff_t ldb_row, 
        const ptrdiff_t ldb_col,
        infiniDtype_t atype,
        infiniDtype_t wtype,
        dim3 blocks,
        dim3 threads,
        void *stream,
        bool has_bias
) {
#define LAUNCH_KERNEL(Tdata, Tweight, Tcompute, BLOCK_SIZE)             \
    rms_norm_gemm_kernel<Tdata, Tweight, Tcompute, BLOCK_SIZE><<<blocks, threads, k * sizeof(Tcompute), (cudaStream_t)stream>>>(          \
        reinterpret_cast<Tdata *>(c),                 \
        reinterpret_cast<const Tdata *>(b),           \
        reinterpret_cast<const Tdata *>(a),           \
        reinterpret_cast<const Tweight *>(w),         \
        reinterpret_cast<const Tdata *>(bias),        \
        m, n, k,                 \
        epsilon,                                            \
        stride_a,                                       \
        ldc,                                            \
        ldb_row,                                        \
        ldb_col,                                        \
        has_bias                                        \
    )                                                                   \

    if (atype == INFINI_DTYPE_F32 && wtype == INFINI_DTYPE_F32) {
        LAUNCH_KERNEL(float, float, float, BLOCK_SIZE);
    } else if (atype == INFINI_DTYPE_F16 && wtype == INFINI_DTYPE_F16) {
        LAUNCH_KERNEL(half, half, float, BLOCK_SIZE);
    } else if (atype == INFINI_DTYPE_F16 && wtype == INFINI_DTYPE_F32) {
        LAUNCH_KERNEL(half, float, float, BLOCK_SIZE);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
#undef LAUNCH_KERNEL

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *c, const void *a, const void *b, const void *w, const void *bias,
    void *stream) const {

    auto &gemm_info = _info.gemm_info;
    auto &rms_norm_info = _info.rms_norm_info;
    
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    
    // ========================================================================
    // 策略选择：根据矩阵大小和数据类型选择最优kernel
    // ========================================================================
    
    // 策略1: 对于适合的尺寸，使用完全融合的单kernel实现
    bool use_fused_kernel = false;
    
    // 重新启用融合kernel
    if (gemm_info.k <= 4096 && gemm_info.m <= 256 && gemm_info.n <= 4096) {
        use_fused_kernel = true;
    }
    
    if (use_fused_kernel) {
        // 使用完全融合的高性能kernel
        if (rms_norm_info.atype == INFINI_DTYPE_F32 && rms_norm_info.wtype == INFINI_DTYPE_F32) {
            // F32融合kernel - 使用超优化版本
            constexpr int BLOCK_M = 16;
            constexpr int BLOCK_N = 64;
            constexpr int BLOCK_K = 32;
            
            dim3 grid((gemm_info.n + BLOCK_N - 1) / BLOCK_N,
                     (gemm_info.m + BLOCK_M - 1) / BLOCK_M);
            dim3 block(256);
            
            size_t shared_size = sizeof(float) * (BLOCK_M + BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N);
            
            rms_norm_gemm_ultra_optimized<BLOCK_M, BLOCK_N, BLOCK_K>
                <<<grid, block, shared_size, cuda_stream>>>(
                    reinterpret_cast<float*>(c),
                    reinterpret_cast<const float*>(a),
                    reinterpret_cast<const float*>(b),
                    reinterpret_cast<const float*>(w),
                    reinterpret_cast<const float*>(bias),
                    gemm_info.m, gemm_info.n, gemm_info.k,
                    rms_norm_info.epsilon,
                    _info.has_bias,
                    rms_norm_info.x_strides[0],
                    gemm_info.c_matrix.row_stride,
                    gemm_info.b_matrix.cols
                );
            
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                // 如果超优化版本失败，尝试简单版本
                dim3 grid_simple(gemm_info.m);
                dim3 block_simple(256);
                size_t shared_simple = gemm_info.k * sizeof(float);
                
                rms_norm_gemm_simple_fused_kernel<float>
                    <<<grid_simple, block_simple, shared_simple, cuda_stream>>>(
                        reinterpret_cast<float*>(c),
                        reinterpret_cast<const float*>(a),
                        reinterpret_cast<const float*>(b),
                        reinterpret_cast<const float*>(w),
                        reinterpret_cast<const float*>(bias),
                        gemm_info.m, gemm_info.n, gemm_info.k,
                        rms_norm_info.epsilon,
                        _info.has_bias,
                        rms_norm_info.x_strides[0],
                        gemm_info.c_matrix.row_stride,
                        gemm_info.b_matrix.cols
                    );
                
                err = cudaGetLastError();
                if (err != cudaSuccess) {
                    goto fallback_implementation;
                }
            }
            
            return INFINI_STATUS_SUCCESS;
            
        } else if (rms_norm_info.atype == INFINI_DTYPE_F16 && rms_norm_info.wtype == INFINI_DTYPE_F16) {
            // F16融合kernel - 使用优化的Tensor Core版本
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            
            if (prop.major >= 7) {  // Volta或更新架构
                constexpr int WMMA_M = 16;
                constexpr int WMMA_N = 16; 
                constexpr int WMMA_K = 16;
                
                dim3 grid((gemm_info.n + WMMA_N - 1) / WMMA_N,
                         (gemm_info.m + WMMA_M - 1) / WMMA_M);
                dim3 block(128);  // 4 warps
                
                size_t shared_size = sizeof(half) * (WMMA_M * gemm_info.k + WMMA_K * WMMA_N) + sizeof(float);
                
                rms_norm_gemm_tensor_core_optimized<WMMA_M, WMMA_N, WMMA_K>
                    <<<grid, block, shared_size, cuda_stream>>>(
                        reinterpret_cast<half*>(c),
                        reinterpret_cast<const half*>(a),
                        reinterpret_cast<const half*>(b),
                        reinterpret_cast<const half*>(w),
                        reinterpret_cast<const half*>(bias),
                        gemm_info.m, gemm_info.n, gemm_info.k,
                        rms_norm_info.epsilon,
                        _info.has_bias,
                        rms_norm_info.x_strides[0],
                        gemm_info.c_matrix.row_stride,
                        gemm_info.b_matrix.cols
                    );
                
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    // 回退到原始实现
                    goto fallback_implementation;
                }
                
                return INFINI_STATUS_SUCCESS;
            }
        }
    }
    
fallback_implementation:
    // ========================================================================
    // 策略2: 原始的两阶段实现（RMSNorm + cuBLAS）
    // 适用于大矩阵或不支持融合kernel的情况
    // ========================================================================
    
    // 只有F32情况下才需要检查workspace大小
    if (rms_norm_info.atype == INFINI_DTYPE_F32 && rms_norm_info.wtype == INFINI_DTYPE_F32) {
        size_t required_workspace = gemm_info.m * gemm_info.k * sizeof(float);
        if (workspace_size < required_workspace) {
            return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
        }
    }
    
    float *normed_buffer_f32 = reinterpret_cast<float*>(workspace);
    half  *normed_buffer_f16 = reinterpret_cast<half*>(workspace);
    
    // ========================================================================
    // 阶段1: 执行优化的RMSNorm，输出到workspace
    // ========================================================================
    // 使用更大的block size以提高占用率和减少同步开销
    constexpr int BLOCK_SIZE = 512;
    dim3 rms_blocks(gemm_info.m);
    dim3 rms_threads(BLOCK_SIZE);
    
    // 分支：F32优化实现；F16优化实现；否则回退
    if (rms_norm_info.atype == INFINI_DTYPE_F32 && rms_norm_info.wtype == INFINI_DTYPE_F32) {
        optimized_rms_norm_kernel_f32<float, float, BLOCK_SIZE><<<rms_blocks, rms_threads, 0, cuda_stream>>>(
            normed_buffer_f32,
            reinterpret_cast<const float*>(a),
            reinterpret_cast<const float*>(w),
            gemm_info.m, gemm_info.k,
            rms_norm_info.epsilon,
            rms_norm_info.x_strides[0]
        );
    } else if (rms_norm_info.atype == INFINI_DTYPE_F16 && rms_norm_info.wtype == INFINI_DTYPE_F16) {
        optimized_rms_norm_kernel_f16<half><<<rms_blocks, rms_threads, 0, cuda_stream>>>(
            normed_buffer_f16,
            reinterpret_cast<const half*>(a),
            reinterpret_cast<const half*>(w),
            gemm_info.m, gemm_info.k,
            rms_norm_info.epsilon,
            rms_norm_info.x_strides[0]
        );
    } else if (rms_norm_info.atype == INFINI_DTYPE_F16 && rms_norm_info.wtype == INFINI_DTYPE_F32) {
        optimized_rms_norm_kernel_f16<float><<<rms_blocks, rms_threads, 0, cuda_stream>>>(
            normed_buffer_f16,
            reinterpret_cast<const half*>(a),
            reinterpret_cast<const float*>(w),
            gemm_info.m, gemm_info.k,
            rms_norm_info.epsilon,
            rms_norm_info.x_strides[0]
        );
    } else {
        // 其他情况（如F16+F32权重）暂时回退到原始实现
        printf("[DEBUG] Using fallback to original kernel for mixed-precision case (a=%d, w=%d)\n", (int)rms_norm_info.atype, (int)rms_norm_info.wtype);

        if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_1024) {
            dim3 threads(CUDA_BLOCK_SIZE_1024);
            launchKernel<CUDA_BLOCK_SIZE_1024>(c, b, a, w, bias,
                gemm_info.m, gemm_info.n, gemm_info.k,
                rms_norm_info.epsilon,
                rms_norm_info.x_strides[0],
                gemm_info.c_matrix.row_stride,
                gemm_info.b_matrix.row_stride, gemm_info.b_matrix.col_stride,
                rms_norm_info.atype, rms_norm_info.wtype,
                dim3(gemm_info.m), threads, stream, _info.has_bias);
        } else {
            dim3 threads(CUDA_BLOCK_SIZE_512);
            launchKernel<CUDA_BLOCK_SIZE_512>(c, b, a, w, bias,
                gemm_info.m, gemm_info.n, gemm_info.k,
                rms_norm_info.epsilon,
                rms_norm_info.x_strides[0],
                gemm_info.c_matrix.row_stride,
                gemm_info.b_matrix.row_stride, gemm_info.b_matrix.col_stride,
                rms_norm_info.atype, rms_norm_info.wtype,
                dim3(gemm_info.m), threads, stream, _info.has_bias);
        }
        return INFINI_STATUS_SUCCESS;
    }
    
    // 检查RMSNorm kernel执行
    cudaError_t kernel_error = cudaGetLastError();
    if (kernel_error != cudaSuccess) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    
    // ========================================================================
    // 阶段2: 使用cuBLAS执行高性能GEMM
    // normed_buffer [m, k] × b [k, n] = c [m, n]
    // ========================================================================
    const float alpha = 1.0f, beta = 0.0f;
    
    CHECK_STATUS(_opaque->internal->useCublas(
        cuda_stream,
        [&](cublasHandle_t handle) {
            // 使用与项目中一致的cublasGemmStridedBatchedEx API
            // 计算: normed_buffer [m, k] × b [k, n] = c [m, n]
            
            // 路径选择：F32或F16（Tensor Core）
            bool use_fp16 = (rms_norm_info.atype == INFINI_DTYPE_F16);
            cudaDataType_t a_type = use_fp16 ? CUDA_R_16F : CUDA_R_32F;
            cudaDataType_t b_type = use_fp16 ? CUDA_R_16F : CUDA_R_32F;
            cudaDataType_t c_type = use_fp16 ? CUDA_R_16F : CUDA_R_32F;
            cudaDataType_t compute_type = CUDA_R_32F; // FP32 accumulate

            // 总是启用Tensor Core（如果可用）以获得最佳性能
            cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
            
            // 设置指针对齐模式以优化内存访问
            cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
            
            // 计算: normed_buffer [m,k] × b [k,n] = c [m,n]
            // 注意：cuBLAS使用列主序，需要正确处理行主序矩阵
            // 对于行主序 C = A*B，在cuBLAS中应该计算 B^T * A^T = C^T
            // 但这里我们要直接计算 A*B，所以：
            // - A: normed_buffer [m,k] 作为第一个矩阵
            // - B: b [k,n] 作为第二个矩阵  
            // - C: c [m,n] 作为输出矩阵
            
            cublasStatus_t status = cublasGemmStridedBatchedEx(
                handle,
                CUBLAS_OP_N,  // B矩阵不转置
                CUBLAS_OP_N,  // A矩阵不转置  
                static_cast<int>(gemm_info.n),     // n: B矩阵的列数，也是输出的列数
                static_cast<int>(gemm_info.m),     // m: A矩阵的行数，也是输出的行数  
                static_cast<int>(gemm_info.k),     // k: 内积维度
                &alpha,
                b,                                 // B矩阵 [k,n]
                b_type,
                static_cast<int>(gemm_info.b_matrix.cols), // ldb: B矩阵的列数作为leading dimension
                0,                                 // strideB
                use_fp16 ? reinterpret_cast<const void*>(normed_buffer_f16) : reinterpret_cast<const void*>(normed_buffer_f32), // A矩阵 [m,k]
                a_type,
                static_cast<int>(gemm_info.k),     // lda: normed_buffer在行主序中的leading dimension是k
                0,                                 // strideA
                &beta,
                c,                                 // C矩阵 [m,n]
                c_type,
                static_cast<int>(gemm_info.c_matrix.ld()),  // ldc: 输出矩阵的leading dimension
                0,                                 // strideC
                1,                                 // batchCount
                compute_type,
                use_fp16 ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_ALGO0
            );
            
            if (status != CUBLAS_STATUS_SUCCESS) {
                return INFINI_STATUS_INTERNAL_ERROR;
            }
            
            return INFINI_STATUS_SUCCESS;
        }
    ));
    
    // ========================================================================
    // 阶段3: 添加bias（如果需要）
    // ========================================================================
    if (_info.has_bias && bias != nullptr) {
        const int threads_per_block = 256;
        const int total_elements = gemm_info.m * gemm_info.n;
        const int blocks_needed = (total_elements + threads_per_block - 1) / threads_per_block;
        
        // 根据输出数据类型调用bias kernel
        if (rms_norm_info.atype == INFINI_DTYPE_F32) {
            add_bias_kernel<float><<<blocks_needed, threads_per_block, 0, cuda_stream>>>(
                reinterpret_cast<float*>(c),
                reinterpret_cast<const float*>(bias),
                gemm_info.m, gemm_info.n,
                gemm_info.c_matrix.row_stride
            );
        } else {
            add_bias_kernel<half><<<blocks_needed, threads_per_block, 0, cuda_stream>>>(
                reinterpret_cast<half*>(c),
                reinterpret_cast<const half*>(bias),
                gemm_info.m, gemm_info.n,
                gemm_info.c_matrix.row_stride
            );
        }
    }
    
    return INFINI_STATUS_SUCCESS;
}
}
