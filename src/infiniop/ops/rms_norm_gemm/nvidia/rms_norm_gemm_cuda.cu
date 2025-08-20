#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "rms_norm_gemm_cuda.cuh"
#include "../cuda/kernel.cuh"
#include <cublas_v2.h>
#include <type_traits>

// ============================================================================
// 高性能融合算子实现：两阶段策略 
// 阶段1: 优化的RMSNorm kernel
// 阶段2: 调用高性能cuBLAS GEMM
// ============================================================================

// RMSNorm kernel - 只做标准化，输出到workspace
template <typename Tdata, typename Tweight, int BLOCK_SIZE>
__global__ void optimized_rms_norm_kernel(
    float *__restrict__ normed_output,      // workspace中的标准化输出
    const Tdata *__restrict__ input,        // 输入数据
    const Tweight *__restrict__ weights,    // RMSNorm权重
    const size_t m, const size_t k,
    const float epsilon,
    const ptrdiff_t input_stride) {
    
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (row >= m) return;
    
    const Tdata *input_row = input + row * input_stride;
    float *output_row = normed_output + row * k;
    
    // 计算sum of squares - 使用高效的block-level reduce
    float thread_sum = 0.0f;
    for (int i = tid; i < k; i += BLOCK_SIZE) {
        float val = static_cast<float>(input_row[i]);
        thread_sum += val * val;
    }
    
    // Block-level reduce using shared memory
    __shared__ float sdata[BLOCK_SIZE];
    sdata[tid] = thread_sum;
    __syncthreads();
    
    // Tree reduction
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // 广播RMS值
    __shared__ float shared_rms;
    if (tid == 0) {
        shared_rms = rsqrtf(sdata[0] / k + epsilon);
    }
    __syncthreads();
    
    // 向量化应用标准化 - 使用float4进行向量化访问
    const int VEC_SIZE = 4;
    const int k_vec = k / VEC_SIZE;
    
    // 处理向量化部分
    for (int i = tid; i < k_vec; i += BLOCK_SIZE) {
        const int base_idx = i * VEC_SIZE;
        if (base_idx + VEC_SIZE <= k) {
            float4 input_vec = reinterpret_cast<const float4*>(input_row)[i];
            float4 weight_vec = reinterpret_cast<const float4*>(weights)[i];
            float4 output_vec;
            
            output_vec.x = input_vec.x * weight_vec.x * shared_rms;
            output_vec.y = input_vec.y * weight_vec.y * shared_rms;
            output_vec.z = input_vec.z * weight_vec.z * shared_rms;
            output_vec.w = input_vec.w * weight_vec.w * shared_rms;
            
            reinterpret_cast<float4*>(output_row)[i] = output_vec;
        }
    }
    
    // 处理剩余元素
    for (int i = k_vec * VEC_SIZE + tid; i < k; i += BLOCK_SIZE) {
        float input_val = static_cast<float>(input_row[i]);
        float weight_val = static_cast<float>(weights[i]);
        output_row[i] = input_val * weight_val * shared_rms;
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
    
    // 计算workspace大小：只有F32情况下才需要额外workspace
    auto info = result.take();
    size_t workspace_size = 0;
    
    // 检查是否是全F32的情况
    if (info.rms_norm_info.atype == INFINI_DTYPE_F32 && info.rms_norm_info.wtype == INFINI_DTYPE_F32) {
        // 需要workspace存储RMSNorm的中间结果
        size_t m = info.gemm_info.m;
        size_t k = info.gemm_info.k;
        workspace_size = m * k * sizeof(float);
    }
    // F16情况下使用原始实现，不需要额外workspace
    
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
    
    // 只有F32情况下才需要检查workspace大小
    if (rms_norm_info.atype == INFINI_DTYPE_F32 && rms_norm_info.wtype == INFINI_DTYPE_F32) {
        size_t required_workspace = gemm_info.m * gemm_info.k * sizeof(float);
        if (workspace_size < required_workspace) {
            return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
        }
    }
    
    float *normed_buffer = reinterpret_cast<float*>(workspace);
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    
    // ========================================================================
    // 阶段1: 执行优化的RMSNorm，输出到workspace
    // ========================================================================
    constexpr int BLOCK_SIZE = 256;
    dim3 rms_blocks(gemm_info.m);
    dim3 rms_threads(BLOCK_SIZE);
    
    // 现在只在全F32的情况下使用优化实现，其他情况回退到原始kernel
    if (rms_norm_info.atype == INFINI_DTYPE_F32 && rms_norm_info.wtype == INFINI_DTYPE_F32) {
        optimized_rms_norm_kernel<float, float, BLOCK_SIZE><<<rms_blocks, rms_threads, 0, cuda_stream>>>(
            normed_buffer,
            reinterpret_cast<const float*>(a),
            reinterpret_cast<const float*>(w),
            gemm_info.m, gemm_info.k,
            rms_norm_info.epsilon,
            rms_norm_info.x_strides[0]
        );
    } else {
        // F16情况下，暂时回退到原始实现
        printf("[DEBUG] Using fallback to original kernel for F16 case\n");
        
        // 使用原始的kernel实现
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
        
        return INFINI_STATUS_SUCCESS;  // 直接返回，不执行后面的cuBLAS部分
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
            
            // 为了简化和兼容性，统一使用F32计算
            // 如果输入是F16，我们已经在RMSNorm阶段转换为F32
            cudaDataType_t a_type = CUDA_R_32F;     // normed_buffer总是float
            cudaDataType_t b_type = CUDA_R_32F;     // 假设权重也转为F32或原本就是F32
            cudaDataType_t c_type = CUDA_R_32F;     // 输出也用F32，后面可以转换
            cudaDataType_t compute_type = CUDA_R_32F;
            
            // 调试信息 - 打印关键参数
            printf("[DEBUG] RMSNormGemm GEMM params: m=%d, n=%d, k=%d\n", 
                static_cast<int>(gemm_info.m), static_cast<int>(gemm_info.n), static_cast<int>(gemm_info.k));
            printf("[DEBUG] Leading dimensions: lda=%d, ldb=%d, ldc=%d\n",
                static_cast<int>(gemm_info.k), 
                static_cast<int>(gemm_info.b_matrix.ld()),
                static_cast<int>(gemm_info.c_matrix.ld()));
            printf("[DEBUG] Data types: a_type=%d, b_type=%d, c_type=%d\n", a_type, b_type, c_type);
            
            cublasStatus_t status = cublasGemmStridedBatchedEx(
                handle,
                CUBLAS_OP_N,  // normed_buffer不转置
                CUBLAS_OP_N,  // b不转置  
                static_cast<int>(gemm_info.n),     // n
                static_cast<int>(gemm_info.m),     // m
                static_cast<int>(gemm_info.k),     // k
                &alpha,
                normed_buffer,                     // A: normed_buffer [m,k]
                a_type,
                static_cast<int>(gemm_info.k),     // lda: normed_buffer的leading dimension
                0,                                 // strideA: 没有batch，stride=0
                b,                                 // B: b [k,n]
                b_type,
                static_cast<int>(gemm_info.b_matrix.ld()),  // ldb: 使用正确的ld()方法
                0,                                 // strideB: 没有batch，stride=0
                &beta,
                c,                                 // C: c [m,n]
                c_type,
                static_cast<int>(gemm_info.c_matrix.ld()),  // ldc: 使用正确的ld()方法
                0,                                 // strideC: 没有batch，stride=0
                1,                                 // batchCount: 只有1个矩阵
                compute_type,
                CUBLAS_GEMM_DEFAULT
            );
            
            printf("[DEBUG] cuBLAS status: %d (0=success)\n", status);
            
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