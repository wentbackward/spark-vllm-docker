import torch, time

SIZE = 8192
DURATION = 5

def bench(label, setup_fn, matmul_fn, duration=DURATION):
    """Run a matmul loop for `duration` seconds, report TFLOPS."""
    try:
        a, b = setup_fn()
        # warmup
        matmul_fn(a, b)
        torch.cuda.synchronize()

        ops = 0
        start = time.perf_counter()
        while time.perf_counter() - start < duration:
            matmul_fn(a, b)
            ops += 1
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        tflops = (2 * SIZE**3 * ops) / elapsed / 1e12
        print(f"{label:6s}: {tflops:7.1f} TFLOPS  ({ops} iters in {elapsed:.1f}s)")
        return tflops
    except Exception as e:
        print(f"{label:6s}: FAILED - {type(e).__name__}: {e}")
        return None

def setup_std(dtype):
    def f():
        a = torch.randn(SIZE, SIZE, device='cuda', dtype=dtype)
        b = torch.randn(SIZE, SIZE, device='cuda', dtype=dtype)
        return a, b
    return f

def setup_fp8():
    # torch._scaled_mm requires B in column-major layout
    a = torch.randn(SIZE, SIZE, device='cuda', dtype=torch.float16).to(torch.float8_e4m3fn)
    b = torch.randn(SIZE, SIZE, device='cuda', dtype=torch.float16).to(torch.float8_e4m3fn).t().contiguous().t()
    return lambda: (a, b)

def matmul_std(a, b):
    return torch.mm(a, b)

def matmul_fp8(a, b):
    scale = torch.tensor(1.0, device='cuda')
    return torch._scaled_mm(a, b, scale_a=scale, scale_b=scale, out_dtype=torch.float16)


# ── NVFP4 (CUTLASS) ─────────────────────────────────────────────────────────
def bench_nvfp4(label="FP4-CUTLASS", duration=DURATION):
    try:
        from vllm import _custom_ops as ops

        # global scales for quantization
        gscale_a = torch.tensor([1.0], device='cuda', dtype=torch.float32)
        gscale_b = torch.tensor([1.0], device='cuda', dtype=torch.float32)

        a_fp16 = torch.randn(SIZE, SIZE, device='cuda', dtype=torch.float16)
        b_fp16 = torch.randn(SIZE, SIZE, device='cuda', dtype=torch.float16)

        a_fp4, scale_a = ops.scaled_fp4_quant(a_fp16, gscale_a)
        b_fp4, scale_b = ops.scaled_fp4_quant(b_fp16, gscale_b)

        alpha = (1.0 / (gscale_a * gscale_b)).to(torch.float32)

        # warmup
        ops.cutlass_scaled_fp4_mm(a_fp4, b_fp4, scale_a, scale_b, alpha, torch.float16)
        torch.cuda.synchronize()

        iters = 0
        start = time.perf_counter()
        while time.perf_counter() - start < duration:
            ops.cutlass_scaled_fp4_mm(a_fp4, b_fp4, scale_a, scale_b, alpha, torch.float16)
            iters += 1
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        tflops = (2 * SIZE**3 * iters) / elapsed / 1e12
        print(f"{label:12s}: {tflops:7.1f} TFLOPS  ({iters} iters in {elapsed:.1f}s)")
        return tflops
    except Exception as e:
        print(f"{label:12s}: FAILED - {type(e).__name__}: {e}")
        return None


# ── NVFP4 via Marlin ────────────────────────────────────────────────────────
def bench_fp4_marlin(label="FP4-MARLIN", duration=DURATION):
    try:
        import os
        os.environ["VLLM_NVFP4_GEMM_BACKEND"] = "marlin"

        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
            is_fp4_marlin_supported,
            rand_marlin_weight_nvfp4_like,
            apply_fp4_marlin_linear,
            marlin_make_workspace_new,
        )

        if not is_fp4_marlin_supported():
            print(f"{label:12s}: not supported by Marlin on this device")
            return None

        # Create a Linear-like weight (N x K) and convert to Marlin NVFP4 format
        group_size = 16  # NVFP4 standard
        weight_bf16 = torch.randn(SIZE, SIZE, device='cuda', dtype=torch.bfloat16)
        w_info = rand_marlin_weight_nvfp4_like(weight_bf16, group_size, input_dtype=torch.bfloat16)
        # Returns: (orig_bf16_ref, marlin_packed_int32, fp8_scale, fp32_global_scale)
        _, weight, weight_scale, weight_global_scale = w_info

        # Input activation (BF16) — MxK
        x = torch.randn(SIZE, SIZE, device='cuda', dtype=torch.bfloat16)
        workspace = marlin_make_workspace_new('cuda')

        # warmup
        apply_fp4_marlin_linear(
            x, weight=weight, weight_scale=weight_scale,
            weight_global_scale=weight_global_scale,
            workspace=workspace, size_n=SIZE, size_k=SIZE,
        )
        torch.cuda.synchronize()

        iters = 0
        start = time.perf_counter()
        while time.perf_counter() - start < duration:
            apply_fp4_marlin_linear(
                x, weight=weight, weight_scale=weight_scale,
                weight_global_scale=weight_global_scale,
                workspace=workspace, size_n=SIZE, size_k=SIZE,
            )
            iters += 1
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        tflops = (2 * SIZE**3 * iters) / elapsed / 1e12
        print(f"{label:12s}: {tflops:7.1f} TFLOPS  ({iters} iters in {elapsed:.1f}s)")
        return tflops
    except Exception as e:
        print(f"{label:12s}: FAILED - {type(e).__name__}: {e}")
        return None


cap = torch.cuda.get_device_capability()
print(f"Device: {torch.cuda.get_device_name()}  (sm_{cap[0]}{cap[1]})")
print(f"Matrix size: {SIZE}x{SIZE},  duration: {DURATION}s per dtype")
print()

bench("FP32", setup_std(torch.float32),  matmul_std)
bench("FP16", setup_std(torch.float16),  matmul_std)
bench("BF16", setup_std(torch.bfloat16), matmul_std)
bench("FP8",  setup_fp8(),               matmul_fp8)

print()
print("── FP4 (vLLM custom kernels) ──")
bench_nvfp4()
bench_fp4_marlin()
