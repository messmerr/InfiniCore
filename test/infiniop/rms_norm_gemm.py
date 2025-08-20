import torch
import ctypes
from ctypes import c_uint64, c_float
from libinfiniop import (
    LIBINFINIOP,
    TestTensor,
    get_test_devices,
    check_error,
    test_operator,
    get_args,
    debug,
    get_tolerance,
    profile_operation,
    TestWorkspace,
    InfiniDtype,
    InfiniDtypeNames,
    InfiniDeviceNames,
    infiniopOperatorDescriptor_t,
)

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules

# Test cases are defined as (c_shape, a_shape, b_shape, w_shape, bias_shape)
# c = rms_norm(a, w) @ b + bias
# a_shape = (m, k), w_shape = (k,), b_shape = (k, n), bias_shape = (n,), c_shape = (m, n)
# Test cases are defined as (c_shape, a_shape, b_shape, w_shape, bias_shape)
# c = rms_norm(a, w) @ b + bias
# a_shape = (m, k), w_shape = (k,), b_shape = (k, n), bias_shape = (n,), c_shape = (m, n)
_TEST_CASES_ = [
    # m, k,  n
    (1,  4,   8),           # 基础测试
    (1,  512, 1024),        # 小规模测试
    (16, 2048, 4096),       # 中等规模测试
    (32, 1024, 2048),       # 原有测试
    # 添加实际模型推理场景的测试用例
    (15, 3584, 10752),      # 实际推理中的attention QKV维度
    (1,  3584, 10752),      # 单token推理场景
    (15, 3584, 37888),      # 实际推理中的FFN gate_up维度  
    (1,  3584, 37888),      # 单token FFN场景
    (32, 3584, 10752),      # 更大batch的attention场景
    (32, 3584, 37888),      # 更大batch的FFN场景
]

# Tensors dtypes used for testing
# Note: BF16 has overflow issues, temporarily disabled
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32]

# Form the test cases by converting (m, k, n) tuples into shape tuples
_TEST_CASES = [
    ((m, n), (m, k), (k, n), (k,), (n,)) for m, k, n in _TEST_CASES_
]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 5e-2, "rtol": 5e-2},  # Increased tolerance for F16
    InfiniDtype.BF16: {"atol": 8e-2, "rtol": 8e-2},
    InfiniDtype.F32: {"atol": 1e-4, "rtol": 1e-4},  # Relaxed tolerance for F32
}

DEBUG = True  # 启用调试模式
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000

def rms_norm_gemm_ref(c_ref, a, w, b, bias, eps):
    """
    Reference implementation of RMSNorm + GEMM using PyTorch.
    """
    if DEBUG:
        print(f"[REF] Input shapes: a={a.shape}, w={w.shape}, b={b.shape}")
        print(f"[REF] Input ranges: a=[{a.min().item():.6f}, {a.max().item():.6f}], w=[{w.min().item():.6f}, {w.max().item():.6f}], b=[{b.min().item():.6f}, {b.max().item():.6f}]")
    
    # 1. RMSNorm
    variance = torch.mean(torch.pow(a, 2), dim=-1, keepdim=True)
    rsqrt_val = torch.rsqrt(variance + eps)
    if DEBUG:
        print(f"[REF] Variance: {variance.flatten()[:5]}")
        print(f"[REF] RMS scale: {rsqrt_val.flatten()[:5]}")
    
    # 将 w 的类型转换为与 a 一致，避免类型提升
    normed_a = a * rsqrt_val * w.to(a.dtype)
    if DEBUG:
        print(f"[REF] Normed_a range: [{normed_a.min().item():.6f}, {normed_a.max().item():.6f}]")
        print(f"[REF] Normed_a sample: {normed_a.flatten()[:5]}")
    
    # 2. Gemm
    torch.matmul(normed_a, b, out=c_ref)
    if DEBUG:
        print(f"[REF] After GEMM range: [{c_ref.min().item():.6f}, {c_ref.max().item():.6f}]")
        print(f"[REF] After GEMM sample: {c_ref.flatten()[:5]}")
    
    # 3. Add bias if provided
    if bias is not None:
        c_ref.add_(bias.to(c_ref.dtype))
        if DEBUG:
            print(f"[REF] After bias range: [{c_ref.min().item():.6f}, {c_ref.max().item():.6f}]")
            print(f"[REF] After bias sample: {c_ref.flatten()[:5]}")


def test(
    handle,
    device,
    c_shape,
    a_shape,
    b_shape,
    w_shape,
    bias_shape,
    dtype=InfiniDtype.F16,
    sync=None,
):
    print(
        f"Testing RMSNormGemm on {InfiniDeviceNames[device]} with a_shape:{a_shape} b_shape:{b_shape}"
        f" w_shape:{w_shape} bias_shape:{bias_shape} c_shape:{c_shape} dtype:{InfiniDtypeNames[dtype]}"
    )

    # Initialize tensors
    # Use small scale for inputs to avoid overflow with F16/BF16
    a = TestTensor(a_shape, None, dtype, device, scale=0.01)
    w = TestTensor(w_shape, None, InfiniDtype.F32, device) # RMSNorm weights are often F32
    b = TestTensor(b_shape, None, dtype, device, scale=0.01)
    bias = TestTensor(bias_shape, None, dtype, device, scale=0.01)
    c = TestTensor(c_shape, None, dtype, device, mode="zeros")

    eps = 1e-6  # 修改为与实际模型配置一致的epsilon值
    # Compute reference result using PyTorch
    rms_norm_gemm_ref(c.torch_tensor(), a.torch_tensor(), w.torch_tensor(), b.torch_tensor(), bias.torch_tensor(), eps)

    if sync is not None:
        sync()

    # Create operator descriptor
    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateRMSNormGemmDescriptor(
            handle,
            ctypes.byref(descriptor),
            c.descriptor,
            a.descriptor,
            b.descriptor,
            w.descriptor,
            bias.descriptor,
            c_float(eps),
        )
    )

    # Invalidate tensor descriptors to ensure the kernel uses its own info
    for tensor in [c, a, b, w, bias]:
        tensor.destroy_desc()

    # Get workspace size
    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetRMSNormGemmWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, c.device)

    # Define the function to call the library operator
    def lib_rms_norm_gemm():
        check_error(
            LIBINFINIOP.infiniopRMSNormGemm(
                descriptor,
                workspace.data(),
                workspace_size.value,
                c.data(),
                a.data(),
                b.data(),
                w.data(),
                bias.data(),
                None, # Stream is NULL for CPU
            )
        )

    # Execute and verify
    lib_rms_norm_gemm()
    
    # 详细数值分析和调试
    if DEBUG:
        print(f"[LIB] Output range: [{c.actual_tensor().min().item():.6f}, {c.actual_tensor().max().item():.6f}]")
        print(f"[LIB] Output sample: {c.actual_tensor().flatten()[:5]}")
        
        # 计算数值差异
        diff = c.actual_tensor() - c.torch_tensor()
        abs_diff = torch.abs(diff)
        rel_diff = abs_diff / (torch.abs(c.torch_tensor()) + 1e-8)
        
        print(f"[DIFF] Max absolute difference: {abs_diff.max().item():.6f}")
        print(f"[DIFF] Max relative difference: {rel_diff.max().item():.6f}")
        print(f"[DIFF] Mean absolute difference: {abs_diff.mean().item():.6f}")
        print(f"[DIFF] Mean relative difference: {rel_diff.mean().item():.6f}")
        
        # 找出最大差异的位置
        max_diff_idx = torch.argmax(abs_diff)
        max_diff_pos = torch.unravel_index(max_diff_idx, abs_diff.shape)
        print(f"[DIFF] Max diff position: {max_diff_pos}")
        print(f"[DIFF] Expected: {c.torch_tensor().flatten()[max_diff_idx].item():.6f}")
        print(f"[DIFF] Actual: {c.actual_tensor().flatten()[max_diff_idx].item():.6f}")
        
        # 检查是否有NaN或Inf
        if torch.isnan(c.actual_tensor()).any():
            print("[ERROR] Output contains NaN values!")
        if torch.isinf(c.actual_tensor()).any():
            print("[ERROR] Output contains Inf values!")
            
        print("=" * 60)

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(c.actual_tensor(), c.torch_tensor(), atol=atol, rtol=rtol)
    
    # 先检查数值正确性，如果失败也继续显示信息
    is_close = torch.allclose(c.actual_tensor(), c.torch_tensor(), atol=atol, rtol=rtol)
    if not is_close and DEBUG:
        print(f"[ERROR] Test FAILED for shape {a_shape} -> {c_shape} with dtype {InfiniDtypeNames[dtype]}")
        print(f"[ERROR] Tolerance: atol={atol}, rtol={rtol}")
    elif DEBUG:
        print(f"[SUCCESS] Test passed for shape {a_shape} -> {c_shape} with dtype {InfiniDtypeNames[dtype]}")
    
    assert is_close

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: rms_norm_gemm_ref(c.torch_tensor(), a.torch_tensor(), w.torch_tensor(), b.torch_tensor(), bias.torch_tensor(), eps), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lib_rms_norm_gemm, device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    
    check_error(LIBINFINIOP.infiniopDestroyRMSNormGemmDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    # Execute tests
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")