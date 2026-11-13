"""Check the Triton projection against the packed PyTorch path on CUDA."""

import torch
from torch.nn import functional as F

from triton_qkv import TritonSplitQKV


def main():
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("highest")
    for shape in ((4, 256, 192), (2, 17, 37)):
        width = shape[-1]
        x = torch.randn(shape, device="cuda", requires_grad=True)
        qk = torch.randn(2 * width, width, device="cuda", requires_grad=True)
        v = torch.randn(width, width, device="cuda", requires_grad=True)
        grad = torch.randn(*shape[:-1], 3 * width, device="cuda")
        for freeze_qk in (False, True):
            qk.requires_grad_(not freeze_qk)
            parameters = (x, v) if freeze_qk else (x, qk, v)
            reference = F.linear(x, torch.cat((qk, v)))
            actual = TritonSplitQKV.apply(x, qk, v)
            torch.testing.assert_close(actual, reference, atol=5e-5, rtol=1e-4)
            expected_grads = torch.autograd.grad(reference, parameters, grad)
            actual_grads = torch.autograd.grad(actual, parameters, grad)
            for got, expected in zip(actual_grads, expected_grads):
                torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-4)
            print(f"{shape}, frozen QK={freeze_qk}: forward and gradients match")


if __name__ == "__main__":
    main()
