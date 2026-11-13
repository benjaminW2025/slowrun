"""Packed QKV output and input gradients without concatenating the weights."""

import torch
import triton
import triton.language as tl


@triton.jit
def _project(X, QK, V, Y, M: tl.constexpr, D: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    group = tl.program_id(2)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(triton.cdiv(D, BK)):
        k = start * BK + inner
        x = tl.load(X + rows[:, None] * D + k[None, :],
                    (rows[:, None] < M) & (k[None, :] < D), other=0)
        if group == 2:
            ptr = V + cols[None, :] * D + k[:, None]
        else:
            ptr = QK + (group * D + cols[None, :]) * D + k[:, None]
        w = tl.load(ptr, (k[:, None] < D) & (cols[None, :] < D), other=0)
        acc = tl.dot(x, w, acc, input_precision="ieee")
    tl.store(Y + rows[:, None] * (3 * D) + group * D + cols[None, :],
             acc, (rows[:, None] < M) & (cols[None, :] < D))


@triton.jit
def _input_grad(G, QK, V, DX, M: tl.constexpr, D: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    # All three projections contribute to dX, even if QK is frozen.
    for group in tl.static_range(3):
        for start in range(triton.cdiv(D, BK)):
            k = start * BK + inner
            g = tl.load(G + rows[:, None] * (3 * D) + group * D + k[None, :],
                        (rows[:, None] < M) & (k[None, :] < D), other=0)
            if group == 2:
                ptr = V + k[:, None] * D + cols[None, :]
            else:
                ptr = QK + (group * D + k[:, None]) * D + cols[None, :]
            w = tl.load(ptr, (k[:, None] < D) & (cols[None, :] < D), other=0)
            acc = tl.dot(g, w, acc, input_precision="ieee")
    tl.store(DX + rows[:, None] * D + cols[None, :],
             acc, (rows[:, None] < M) & (cols[None, :] < D))


class TritonSplitQKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, qk, v):
        width = x.size(-1)
        x = x.contiguous()
        rows = x.numel() // width
        output = x.new_empty((*x.shape[:-1], 3 * width))
        tiles = (triton.cdiv(rows, 32), triton.cdiv(width, 32))
        with torch.cuda.device(x.device):
            _project[(*tiles, 3)](x, qk, v, output, rows, width, 32, 32, 32)
        ctx.save_for_backward(x, qk, v)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, qk, v = ctx.saved_tensors
        width = x.size(-1)
        grad = grad_output.contiguous().view(-1, 3 * width)
        grad_input = grad_qk = grad_v = None
        if ctx.needs_input_grad[0]:
            grad_input = torch.empty_like(x)
            tiles = (triton.cdiv(grad.size(0), 32), triton.cdiv(width, 32))
            with torch.cuda.device(x.device):
                _input_grad[tiles](grad, qk, v, grad_input, grad.size(0), width, 32, 32, 32)
        inputs = x.view(-1, width)
        if ctx.needs_input_grad[1]:
            grad_qk = grad[:, :2 * width].T @ inputs
        if ctx.needs_input_grad[2]:
            grad_v = grad[:, 2 * width:].T @ inputs
        return grad_input, grad_qk, grad_v
