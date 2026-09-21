"""Small GPU acceptance checks for the optional Triton backward.

Run from the assignment root:
    uv run python -m cs336_systems.check_flash_backward
For numerical comparison only (no adapter changes or autograd checks):
    uv run python -m cs336_systems.check_flash_backward --direct-only --dtypes float32 --lengths 128

Defaults: batch=4, d=64, L=128 and 16, FP32/BF16, causal=False/True.
Checks use the actual saved O/L for a direct backward comparison, then use
autograd and a dense reference. A call spy verifies that the optional backward
ran; calling the required compiled backward in the Full path raises an error.
This supplements, rather than replaces, the supplied assignment pytest tests.
--direct-only returns after comparing the two backward functions on identical
saved tensors; it does not call FlashAttentionTritonFull.apply or .backward().
"""

import argparse
from unittest.mock import patch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtypes", nargs="+", choices=["float32", "bfloat16"],
                        default=["float32", "bfloat16"])
    parser.add_argument("--lengths", nargs="+", type=int, default=[128, 16])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--direct-only", action="store_true",
                        help="Only compare backward functions on identical saved tensors; skip autograd checks")
    args = parser.parse_args()
    if any(length < 1 for length in args.lengths):
        parser.error("lengths must be positive")
    return args


class CaptureContext:
    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors


def check_case(torch, required, optional, length, dtype, causal, direct_only=False):
    shape = (4, length, 64)
    Q, K, V, dO = [
        torch.randn(shape, device="cuda", dtype=dtype)
        for _ in range(4)
    ]

    # Explicit supplemental tolerances; these do not change course tests.
    atol, rtol = (1e-4, 1e-3) if dtype == torch.float32 else (2e-2, 2e-2)

    def compare(actual, expected, name):
        if not bool(torch.isfinite(actual).all()):
            raise AssertionError(f"{name}: non-finite output")
        try:
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        except AssertionError as exc:
            raise AssertionError(f"{name}: {exc}") from exc
        return float((actual.float() - expected.float()).abs().max())

    # Directly invoke the inherited forward to capture its actual saved tensors.
    # Both backward implementations receive the EXACT same Q/K/V/O/dO/L.
    with torch.no_grad():
        ctx = CaptureContext()
        required.FlashAttentionTriton.forward(ctx, Q, K, V, causal)
        L, saved_Q, saved_K, saved_V, O = ctx.saved_tensors
        dense_grads = required.flash_backward_pytorch(
            saved_Q, saved_K, saved_V, O, dO, L, causal
        )
        tiled_grads = optional.flash_backward_triton(
            saved_Q, saved_K, saved_V, O, dO, L, causal
        )
        torch.cuda.synchronize()
        direct_errors = [
            compare(actual, expected, f"direct {name}")
            for name, actual, expected in zip(("dQ", "dK", "dV"), tiled_grads, dense_grads)
        ]

    del dense_grads, tiled_grads, ctx, O, L

    if direct_only:
        print(
            f"DIRECT PASS B=4 L={length} d=64 dtype={dtype} causal={causal}\n"
            f"  max_abs(dQ,dK,dV)={direct_errors}\n"
            f"  atol={atol}, rtol={rtol}",
            flush=True,
        )
        return

    # Independent reference: FP32 score/softmax/value arithmetic, output cast to
    # the input dtype. This path uses ordinary autograd, not either custom backward.
    ref_inputs = [x.detach().clone().requires_grad_(True) for x in (Q, K, V)]
    q_ref, k_ref, v_ref = ref_inputs
    scores = (q_ref.float() @ k_ref.float().transpose(-2, -1)) / (64 ** 0.5)
    if causal:
        indices = torch.arange(length, device="cuda")
        allowed = indices[:, None] >= indices[None, :]
        scores = scores + torch.where(allowed, 0.0, -1e6)
    ref_output = (torch.softmax(scores, dim=-1) @ v_ref.float()).to(dtype)
    ref_output.backward(dO)

    full_inputs = [x.detach().clone().requires_grad_(True) for x in (Q, K, V)]
    actual_backward = optional.flash_backward_triton
    with patch.object(optional, "flash_backward_triton", wraps=actual_backward) as spy:
        with patch.object(
            required, "flash_backward_pytorch",
            side_effect=AssertionError("Full path called the required dense backward"),
        ):
            # False tests the omitted optional argument; True tests four inputs.
            if causal:
                output = optional.FlashAttentionTritonFull.apply(*full_inputs, True)
            else:
                output = optional.FlashAttentionTritonFull.apply(*full_inputs)
            output.backward(dO)
            torch.cuda.synchronize()
        if spy.call_count != 1:
            raise AssertionError(f"Expected one optional backward call, got {spy.call_count}")

    output_error = compare(output, ref_output, "autograd O")
    grad_errors = [
        compare(actual.grad, expected.grad, f"autograd {name}")
        for name, actual, expected in zip(("dQ", "dK", "dV"), full_inputs, ref_inputs)
    ]
    print(
        f"PASS B=4 L={length} d=64 dtype={dtype} causal={causal}\n"
        f"  direct max_abs(dQ,dK,dV)={direct_errors}\n"
        f"  autograd max_abs(O)={output_error}; grads={grad_errors}\n"
        f"  optional backward calls=1; atol={atol}, rtol={rtol}",
        flush=True,
    )


def main():
    args = parse_args()
    import torch
    import cs336_systems.flash_attention as required
    import cs336_systems.flash_attention_triton_backward as optional

    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required; these are real GPU checks")
    if "bfloat16" in args.dtypes and not torch.cuda.is_bf16_supported():
        raise SystemExit("BF16 is unsupported; use --dtypes float32 for an FP32-only check")
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(args.seed)
    print(f"GPU: {torch.cuda.get_device_name()}; PyTorch: {torch.__version__}", flush=True)
    for dtype_name in args.dtypes:
        for length in args.lengths:
            for causal in (False, True):
                print(f"Checking {dtype_name}, L={length}, causal={causal} ...", flush=True)
                check_case(
                    torch, required, optional, length, getattr(torch, dtype_name),
                    causal, direct_only=args.direct_only,
                )
    if args.direct_only:
        print("All requested direct numerical comparisons passed; autograd/adapter paths were not tested.")
    else:
        print("All requested small-case checks passed.")


if __name__ == "__main__":
    main()
