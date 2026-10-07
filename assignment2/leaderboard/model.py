"""4-GPU leaderboard LM: data parallel 2 x tensor parallel 2.

Each data-parallel replica owns half of the batch. Tensor parallel splits
heads, SwiGLU, and the LM head across the other two devices. The training
loss is vocab-parallel cross entropy, so full logits are not gathered onto
one device. LeaderboardAdamW sums the matching shards before the update.
"""

from __future__ import annotations

import math
import sys
import threading
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cs336-basics"))

from cs336_basics.model import RMSNorm, RotaryEmbedding, silu  # noqa: E402
from cs336_basics.nn_utils import softmax  # noqa: E402
from cs336_basics.optimizer import AdamW  # noqa: E402


_profile = False
_events = []


def set_profile(enabled):
    global _profile
    _profile = enabled
    _events.clear()


class _CudaSpan:
    """Records GPU time on each device. critical_ms is the busiest device."""

    def __init__(self, name, devices):
        self.name = name
        self.devices = []
        self.starts = []
        if not _profile:
            return
        for device in devices:
            device = torch.device(device)
            if device.type == "cuda":
                self.devices.append(device)

    def __enter__(self):
        for device in self.devices:
            with torch.cuda.device(device):
                event = torch.cuda.Event(True)
                event.record()
                self.starts.append((device, event))
        return self

    def __exit__(self, *exc):
        for device, start in self.starts:
            with torch.cuda.device(device):
                end = torch.cuda.Event(True)
                end.record()
            _events.append((self.name, device.index, start, end))


def profile_report():
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(index)
    per_name = {}
    for name, index, start, end in _events:
        per_name.setdefault(name, {})
        per_name[name][index] = per_name[name].get(index, 0.0) + start.elapsed_time(end)
    lines = []
    for name, devices in sorted(per_name.items()):
        critical = max(devices.values())
        detail = " ".join(f"cuda{index}:{ms / 1000:.3f}s" for index, ms in sorted(devices.items()))
        lines.append(f"{name} critical_s={critical / 1000:.3f} {detail}")
    _events.clear()
    return "\n".join(lines)


def _sync_cuda():
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(index)


def _to_device(value, device, dtype=None):
    """Differentiable copy. Clone when .to() would return the same storage."""
    moved = value.to(device=device, dtype=value.dtype if dtype is None else dtype)
    if moved.data_ptr() == value.data_ptr():
        return value.clone()
    return moved


def sum_across(left, right):
    """Sum a tensor-parallel pair and return the result on both devices."""
    total = left + _to_device(right, left.device, left.dtype)
    return total, _to_device(total, right.device, right.dtype)


def sync_dp_grads(pairs):
    """Give every data-parallel copy the sum of the two microbatch gradients."""
    for left, right in pairs:
        if left.grad is None and right.grad is None:
            continue
        if left.grad is None or right.grad is None:
            raise RuntimeError("data-parallel pair is missing a gradient")
        total = left.grad + right.grad.to(device=left.device, dtype=left.dtype)
        left.grad = total
        right.grad = total.to(device=right.device, dtype=right.dtype)


class LeaderboardAdamW(AdamW):
    """cs336 AdamW. Data-parallel shards are summed immediately before the update."""

    def __init__(self, params, dp_pairs, **kwargs):
        super().__init__(params, **kwargs)
        self.dp_pairs = list(dp_pairs)

    def step(self, closure=None):
        sync_dp_grads(self.dp_pairs)
        return super().step(closure)


def _trunc_normal(tensor, std):
    return nn.init.trunc_normal_(tensor, std=std, a=-3 * std, b=3 * std)


def _linear_param(d_out, d_in, device, dtype, fan_in, fan_out):
    std = math.sqrt(2 / (fan_in + fan_out))
    weight = _trunc_normal(torch.empty(d_out, d_in), std)
    return nn.Parameter(weight.to(device=device, dtype=dtype))


def _copy_param(param, source):
    param.copy_(source.detach().to(device=param.device, dtype=param.dtype))


def _token_nll(rows0, rows1, index):
    """Per-token NLL. rows are (tokens, vocab/2) on the tensor-parallel pair."""
    half = rows0.shape[-1]
    scores0 = rows0.float()
    scores1 = rows1.float()
    maximum = torch.maximum(scores0.amax(dim=-1), scores1.amax(dim=-1).to(scores0.device))
    exp0 = torch.exp(scores0 - maximum[:, None])
    exp1 = torch.exp(scores1 - maximum.to(scores1.device)[:, None])
    normalizer = exp0.sum(dim=-1) + exp1.sum(dim=-1).to(scores0.device)
    logsumexp = maximum + torch.log(normalizer)
    on_first = index < half
    positions0 = torch.arange(index.shape[0], device=scores0.device)
    positions1 = torch.arange(index.shape[0], device=scores1.device)
    chosen0 = scores0[positions0, index.clamp(max=half - 1)]
    chosen1 = scores1[positions1, (index - half).clamp(min=0, max=half - 1).to(scores1.device)].to(scores0.device)
    chosen = torch.where(on_first, chosen0, chosen1)
    return -chosen + logsumexp


def _token_grad(rows0, rows1, index, scale):
    """Gradient of the summed NLL. scale is d(loss)/d(sum)."""
    half = rows0.shape[-1]
    scores0 = rows0.float()
    scores1 = rows1.float()
    maximum = torch.maximum(scores0.amax(dim=-1), scores1.amax(dim=-1).to(scores0.device))
    exp0 = torch.exp(scores0 - maximum[:, None])
    exp1 = torch.exp(scores1 - maximum.to(scores1.device)[:, None])
    normalizer = exp0.sum(dim=-1) + exp1.sum(dim=-1).to(scores0.device)
    scale1 = scale.to(scores1.device)
    grad0 = exp0 / normalizer[:, None] * scale
    grad1 = exp1 / normalizer.to(exp1.device)[:, None] * scale1
    on_first = index.to(scores0.device) < half
    positions0 = torch.arange(index.shape[0], device=scores0.device)
    first = positions0[on_first]
    if first.numel():
        grad0[first, index.to(scores0.device)[on_first]] -= scale
    on_second = ~on_first
    positions1 = torch.arange(index.shape[0], device=scores1.device)
    second = positions1[on_second.to(scores1.device)]
    if second.numel():
        grad1[second, (index.to(scores1.device) - half)[on_second.to(scores1.device)]] -= scale1
    return grad0, grad1


class ShardedCrossEntropy(torch.autograd.Function):
    """Sum of token NLLs. The vocab stays split across the tensor-parallel pair."""

    @staticmethod
    def forward(ctx, shard0, shard1, targets):
        half = shard0.shape[-1]
        rows0 = shard0.reshape(-1, half)
        rows1 = shard1.reshape(-1, half)
        index = targets.reshape(-1).to(device=shard0.device, dtype=torch.long)
        total = shard0.new_zeros((), dtype=torch.float32)
        with _CudaSpan("cross_entropy_fwd", (shard0.device, shard1.device)):
            for start in range(0, rows0.shape[0], 512):
                end = min(start + 512, rows0.shape[0])
                total = total + _token_nll(rows0[start:end], rows1[start:end], index[start:end]).sum()
        ctx.save_for_backward(shard0, shard1, index.view_as(targets))
        return total

    @staticmethod
    def backward(ctx, grad_output):
        shard0, shard1, targets = ctx.saved_tensors
        half = shard0.shape[-1]
        grad0 = torch.empty_like(shard0)
        grad1 = torch.empty_like(shard1)
        flat0 = grad0.reshape(-1, half)
        flat1 = grad1.reshape(-1, half)
        rows0 = shard0.reshape(-1, half)
        rows1 = shard1.reshape(-1, half)
        index = targets.reshape(-1)
        scale = grad_output.to(dtype=torch.float32)
        with _CudaSpan("cross_entropy_bwd", (shard0.device, shard1.device)):
            for start in range(0, rows0.shape[0], 512):
                end = min(start + 512, rows0.shape[0])
                piece0, piece1 = _token_grad(rows0[start:end], rows1[start:end], index[start:end], scale)
                flat0[start:end] = piece0.to(dtype=shard0.dtype)
                flat1[start:end] = piece1.to(dtype=shard1.dtype)
        return grad0, grad1, None


class ColumnParallel(nn.Module):
    """Split d_out across the tensor-parallel pair. The input stays duplicated."""

    def __init__(self, d_in, d_out, devices, dtype):
        super().__init__()
        if d_out % 2 != 0:
            raise ValueError(f"d_out={d_out} is not divisible by tensor parallel size 2")
        local = d_out // 2
        self.weight0 = _linear_param(local, d_in, devices[0], dtype, d_in, d_out)
        self.weight1 = _linear_param(local, d_in, devices[1], dtype, d_in, d_out)

    def forward(self, x0, x1):
        return F.linear(x0, self.weight0), F.linear(x1, self.weight1)

    @torch.no_grad()
    def load_rows(self, weight):
        top, bottom = weight.chunk(2, dim=0)
        self.weight0.copy_(top.to(device=self.weight0.device, dtype=self.weight0.dtype))
        self.weight1.copy_(bottom.to(device=self.weight1.device, dtype=self.weight1.dtype))


class RowParallel(nn.Module):
    """Split d_in across the pair and sum the partial outputs."""

    def __init__(self, d_in, d_out, devices, dtype):
        super().__init__()
        if d_in % 2 != 0:
            raise ValueError(f"d_in={d_in} is not divisible by tensor parallel size 2")
        local = d_in // 2
        self.weight0 = _linear_param(d_out, local, devices[0], dtype, d_in, d_out)
        self.weight1 = _linear_param(d_out, local, devices[1], dtype, d_in, d_out)

    def forward(self, x0, x1):
        return sum_across(F.linear(x0, self.weight0), F.linear(x1, self.weight1))

    @torch.no_grad()
    def load_cols(self, weight):
        left, right = weight.chunk(2, dim=1)
        self.weight0.copy_(left.to(device=self.weight0.device, dtype=self.weight0.dtype))
        self.weight1.copy_(right.to(device=self.weight1.device, dtype=self.weight1.dtype))


class CausalAttention(nn.Module):
    def __init__(self, d_model, num_heads, context_length, devices, dtype, attention):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} is not divisible by num_heads={num_heads}")
        if num_heads % 2 != 0:
            raise ValueError(f"num_heads={num_heads} is not divisible by tensor parallel size 2")
        self.d_head = d_model // num_heads
        self.local_heads = num_heads // 2
        self.attention = attention
        self.q = ColumnParallel(d_model, d_model, devices, dtype)
        self.k = ColumnParallel(d_model, d_model, devices, dtype)
        self.v = ColumnParallel(d_model, d_model, devices, dtype)
        self.o = RowParallel(d_model, d_model, devices, dtype)
        self.rope0 = RotaryEmbedding(context_length, self.d_head).to(devices[0])
        self.rope1 = RotaryEmbedding(context_length, self.d_head).to(devices[1])

    def forward(self, x0, x1):
        q0, q1 = self.q(x0, x1)
        k0, k1 = self.k(x0, x1)
        v0, v1 = self.v(x0, x1)
        h0 = _attend(self, q0, k0, v0, self.rope0)
        h1 = _attend(self, q1, k1, v1, self.rope1)
        return self.o(h0, h1)


class SwiGLUParallel(nn.Module):
    def __init__(self, d_model, d_ff, devices, dtype):
        super().__init__()
        if d_ff % 2 != 0:
            raise ValueError(f"d_ff={d_ff} is not divisible by tensor parallel size 2")
        self.w1 = ColumnParallel(d_model, d_ff, devices, dtype)
        self.w3 = ColumnParallel(d_model, d_ff, devices, dtype)
        self.w2 = RowParallel(d_ff, d_model, devices, dtype)

    def forward(self, x0, x1):
        gate0, gate1 = self.w1(x0, x1)
        up0, up1 = self.w3(x0, x1)
        return self.w2(silu(gate0) * up0, silu(gate1) * up1)


class TransformerBlockParallel(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, context_length, devices, dtype, attention):
        super().__init__()
        self.ln1_0 = RMSNorm(d_model, device=devices[0]).to(dtype)
        self.ln1_1 = RMSNorm(d_model, device=devices[1]).to(dtype)
        self.attn = CausalAttention(d_model, num_heads, context_length, devices, dtype, attention)
        self.ln2_0 = RMSNorm(d_model, device=devices[0]).to(dtype)
        self.ln2_1 = RMSNorm(d_model, device=devices[1]).to(dtype)
        self.ffn = SwiGLUParallel(d_model, d_ff, devices, dtype)

    def forward(self, x0, x1):
        attn0, attn1 = self.attn(self.ln1_0(x0), self.ln1_1(x1))
        hidden0 = x0 + attn0
        hidden1 = x1 + attn1
        ffn0, ffn1 = self.ffn(self.ln2_0(hidden0), self.ln2_1(hidden1))
        return hidden0 + ffn0, hidden1 + ffn1


class Replica(nn.Module):
    """One data-parallel copy on a tensor-parallel device pair."""

    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, devices, dtype, attention, checkpoint_activations, compile_blocks=False):
        super().__init__()
        self.devices = tuple(devices)
        self.checkpoint_activations = checkpoint_activations
        embed = _trunc_normal(torch.empty(vocab_size, d_model), 1.0)
        self.embed = nn.Parameter(embed.to(device=devices[0], dtype=dtype))
        self.blocks = nn.ModuleList(
            [
                TransformerBlockParallel(
                    d_model, num_heads, d_ff, context_length, devices, dtype, attention
                )
                for _ in range(num_layers)
            ]
        )
        if compile_blocks:
            self.blocks = nn.ModuleList(torch.compile(block, dynamic=False) for block in self.blocks)
        self.ln0 = RMSNorm(d_model, device=devices[0]).to(dtype)
        self.ln1 = RMSNorm(d_model, device=devices[1]).to(dtype)
        self.head = ColumnParallel(d_model, vocab_size, devices, dtype)

    def _logit_shards(self, token_ids):
        hidden0 = self.embed[token_ids.to(self.devices[0])]
        hidden1 = _to_device(hidden0, self.devices[1])
        for block in self.blocks:
            if self.checkpoint_activations:
                hidden0, hidden1 = checkpoint(block, hidden0, hidden1, use_reentrant=True)
            else:
                hidden0, hidden1 = block(hidden0, hidden1)
        return self.head(self.ln0(hidden0), self.ln1(hidden1))

    def forward(self, token_ids):
        part0, part1 = self._logit_shards(token_ids)
        return torch.cat((part0, part1.to(part0.device)), dim=-1)

    def nll_sum(self, token_ids, targets):
        part0, part1 = self._logit_shards(token_ids)
        return ShardedCrossEntropy.apply(part0, part1, targets)

    @torch.no_grad()
    def copy_from_basics(self, source):
        _copy_param(self.embed, source.token_embeddings.weight)
        for block, src in zip(self.blocks, source.layers):
            _copy_param(block.ln1_0.weight, src.ln1.weight)
            _copy_param(block.ln1_1.weight, src.ln1.weight)
            block.attn.q.load_rows(src.attn.q_proj.weight)
            block.attn.k.load_rows(src.attn.k_proj.weight)
            block.attn.v.load_rows(src.attn.v_proj.weight)
            block.attn.o.load_cols(src.attn.output_proj.weight)
            _copy_param(block.ln2_0.weight, src.ln2.weight)
            _copy_param(block.ln2_1.weight, src.ln2.weight)
            block.ffn.w1.load_rows(src.ffn.w1.weight)
            block.ffn.w3.load_rows(src.ffn.w3.weight)
            block.ffn.w2.load_cols(src.ffn.w2.weight)
        _copy_param(self.ln0.weight, source.ln_final.weight)
        _copy_param(self.ln1.weight, source.ln_final.weight)
        self.head.load_rows(source.lm_head.weight)


class LeaderboardLM(nn.Module):
    """Two devices are one tensor-parallel pair. Four devices add data parallel."""

    def __init__(
        self,
        vocab_size,
        context_length,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        devices,
        dtype=torch.float32,
        attention="triton",
        checkpoint_activations=True,
        compile_blocks=False,
    ):
        super().__init__()
        if len(devices) not in (2, 4):
            raise ValueError("expected 2 devices (tensor parallel) or 4 (data parallel 2 x tensor parallel 2)")
        if vocab_size % 2 != 0:
            raise ValueError(f"vocab_size={vocab_size} is not divisible by tensor parallel size 2")
        self.devices = tuple(torch.device(device) for device in devices)
        self.attention = attention
        common = dict(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            dtype=dtype,
            attention=attention,
            checkpoint_activations=checkpoint_activations,
            compile_blocks=compile_blocks,
        )
        self.replica0 = Replica(**common, devices=self.devices[:2])
        self.replica1 = Replica(**common, devices=self.devices[2:]) if len(self.devices) == 4 else None
        self.dp_pairs = []
        if self.replica1 is not None:
            self.dp_pairs = list(zip(self.replica0.parameters(), self.replica1.parameters()))
            with torch.no_grad():
                for src, dst in self.dp_pairs:
                    dst.copy_(src.to(device=dst.device, dtype=dst.dtype))

    def _split_batch(self, token_ids):
        if token_ids.shape[0] % 2 != 0:
            raise ValueError("batch size must be divisible by data parallel size 2")
        if self.attention == "triton" and token_ids.shape[-1] % 16 != 0:
            raise ValueError("Triton FlashAttention requires a sequence length multiple of 16")
        half = token_ids.shape[0] // 2
        return half

    def _replicas(self, token_ids, targets):
        if self.replica1 is None:
            if self.attention == "triton" and token_ids.shape[-1] % 16 != 0:
                raise ValueError("Triton FlashAttention requires a sequence length multiple of 16")
            return ((self.replica0, token_ids, targets),)
        half = self._split_batch(token_ids)
        return (
            (self.replica0, token_ids[:half], targets[:half]),
            (self.replica1, token_ids[half:], targets[half:]),
        )

    def forward(self, token_ids):
        parts = self._replicas(token_ids, token_ids)
        if len(parts) == 1:
            replica, tokens, _ = parts[0]
            return replica(tokens)
        logits = [replica(tokens) for replica, tokens, _ in parts]
        return torch.cat((logits[0], logits[1].to(logits[0].device)), dim=0)

    def nll(self, token_ids, targets):
        """Mean token NLL, matching cs336_basics.cross_entropy(...).sum()."""
        losses = [replica.nll_sum(tokens, replica_targets) for replica, tokens, replica_targets in self._replicas(token_ids, targets)]
        total = losses[0]
        for loss in losses[1:]:
            total = total + loss.to(total.device)
        return total / targets.numel()

    def nll_backward(self, token_ids, targets):
        """Same mean NLL as nll(). Two data-parallel replicas run together."""
        specs = self._replicas(token_ids, targets)
        losses = [None] * len(specs)
        errors = []

        def run(index, replica, tokens, replica_targets):
            try:
                if replica.devices[0].type == "cuda":
                    torch.cuda.set_device(replica.devices[0])
                losses[index] = replica.nll_sum(tokens, replica_targets)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=run, args=(index, *spec)) for index, spec in enumerate(specs)]
        if _profile:
            _sync_cuda()
            started = time.perf_counter()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if _profile:
            _sync_cuda()
            forward_s = time.perf_counter() - started
        if errors:
            raise errors[0]

        count = targets.numel()
        errors.clear()

        def backward(loss):
            try:
                if loss.device.type == "cuda":
                    torch.cuda.set_device(loss.device)
                loss.backward(gradient=loss.new_full((), 1.0 / count))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=backward, args=(loss,)) for loss in losses]
        if _profile:
            started = time.perf_counter()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if _profile:
            _sync_cuda()
            self.phase_s = {"forward": forward_s, "backward": time.perf_counter() - started}
        if errors:
            raise errors[0]
        total = losses[0].detach()
        for loss in losses[1:]:
            total = total + loss.detach().to(total.device)
        return total / count

    def sync_dp_grads(self):
        sync_dp_grads(self.dp_pairs)

    @torch.no_grad()
    def copy_from_basics(self, source):
        self.replica0.copy_from_basics(source)
        if self.replica1 is not None:
            self.replica1.copy_from_basics(source)


def pytorch_causal_attention(q, k, v):
    """Same math as cs336_basics scaled dot-product attention, without an LxL mask tensor from the caller."""
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    length = q.shape[-2]
    positions = torch.arange(length, device=q.device)
    allowed = positions[:, None] >= positions[None, :]
    scores = torch.where(allowed, scores, float("-inf"))
    return softmax(scores, dim=-1) @ v


def _attend(module, q, k, v, rope):
    batch, seq, _ = q.shape
    q = q.view(batch, seq, module.local_heads, module.d_head).permute(0, 2, 1, 3)
    k = k.view(batch, seq, module.local_heads, module.d_head).permute(0, 2, 1, 3)
    v = v.view(batch, seq, module.local_heads, module.d_head).permute(0, 2, 1, 3)
    dtype = q.dtype
    q = rope(q, None).to(dtype)
    k = rope(k, None).to(dtype)
    heads = module.local_heads
    if module.attention == "sdpa":
        out = _sdpa_causal(q, k, v)
    else:
        flat_q = q.reshape(batch * heads, seq, module.d_head)
        flat_k = k.reshape(batch * heads, seq, module.d_head)
        flat_v = v.reshape(batch * heads, seq, module.d_head)
        if module.attention == "pytorch":
            out = pytorch_causal_attention(flat_q, flat_k, flat_v)
        elif module.attention == "triton":
            out = _flash(flat_q, flat_k, flat_v)
        else:
            raise ValueError(f"unknown attention backend {module.attention!r}")
        out = out.view(batch, heads, seq, module.d_head)
    out = out.permute(0, 2, 1, 3)
    return out.reshape(batch, seq, heads * module.d_head).contiguous()


def _sdpa_causal(q, k, v):
    """Blackwell FlashAttention. q, k, v are (batch, heads, sequence, dim)."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    with _CudaSpan("attention_fwd", (q.device,)):
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def _flash(q, k, v):
    from cs336_systems.flash_attention_triton_backward import FlashAttentionTritonFull

    with torch.cuda.device(q.device):
        return FlashAttentionTritonFull.apply(q, k, v, True)
