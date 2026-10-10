"""Serial data-parallel A/B episodes for the RTX 8000 Slowrun experiment.

The returned parameter gradients equal the mean of independent virtual-rank
gradients. No optimizer step occurs until every trial has been restored.
"""

import torch


TRIAL_PATTERNS = {
    "off": (),
    "mlp": (".mlp.",),
    "qk": (".attn.c_q.", ".attn.c_k."),
    "v": (".attn.c_v.",),
}


def select_trial_parameters(named_parameters, muon_parameter_ids, mode):
    if mode not in TRIAL_PATTERNS:
        raise ValueError(f"Unknown temporary-step mode: {mode}")
    selected = [p for name, p in named_parameters
                if id(p) in muon_parameter_ids
                and any(pattern in name for pattern in TRIAL_PATTERNS[mode])]
    if mode != "off" and not selected:
        raise ValueError(f"No Muon matrices selected for temporary-step mode {mode}")
    return selected


@torch.no_grad()
def _temporary_step(trial_params, step_norm):
    original_weights = [p.detach().clone() for p in trial_params]
    direction = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
                 for p in trial_params]
    norm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(direction)))
    if not bool(torch.isfinite(norm)):
        raise FloatingPointError(f"Invalid temporary A-gradient norm: {norm.item()}")
    if norm.item() == 0:
        # Zero-initialized attention output projections block Q/K/V gradients
        # on the first optimizer step. There is no direction to normalize yet.
        return None
    alpha = -step_norm / norm.item()
    torch._foreach_add_(trial_params, direction, alpha=alpha)
    return original_weights


def collect_virtual_gradients(
    model,
    trainable_params,
    trial_params,
    x,
    y,
    next_batch,
    forward_loss,
    scaler,
    virtual_ranks,
    microsteps_per_half,
    step_norm,
    capture_episode=None,
):
    """Run rank-local A/B trials and leave their averaged gradient in p.grad.

    Each virtual rank gets ``microsteps_per_half`` batches for A and the same
    number for B. ``forward_loss`` must return a scalar mean loss. GradScaler's
    common scale cancels when the A gradient is normalized; gradients are
    unscaled by the caller before the real optimizer step. Normalize the
    accumulated parameter gradients after backward, in FP32: dividing each
    loss first can underflow non-target softmax gradients in FP16.
    """
    if virtual_ranks < 1 or microsteps_per_half < 1:
        raise ValueError("Each virtual rank needs at least one A and one B microbatch")
    if step_norm <= 0:
        raise ValueError("Temporary step norm must be positive")
    total_microsteps = 2 * virtual_ranks * microsteps_per_half
    accumulated_grads = [None] * len(trainable_params)
    train_loss = x.new_zeros((), dtype=torch.float32)
    epoch = None

    for virtual_rank in range(virtual_ranks):
        model.zero_grad(set_to_none=True)
        original_weights = None
        try:
            for half in range(2):
                for _ in range(microsteps_per_half):
                    if capture_episode is not None and virtual_rank == 0:
                        capture_episode[half].append((x.detach(), y.detach()))
                    loss = forward_loss(x, y)
                    train_loss = train_loss + loss.detach() / total_microsteps
                    scaler.scale(loss).backward()
                    x, y, epoch = next_batch()
                if half == 0 and trial_params:
                    original_weights = _temporary_step(trial_params, step_norm)
        finally:
            if original_weights is not None:
                with torch.no_grad():
                    torch._foreach_copy_(trial_params, original_weights)

        for i, p in enumerate(trainable_params):
            if p.grad is not None:
                if accumulated_grads[i] is None:
                    accumulated_grads[i] = p.grad.detach().clone()
                else:
                    accumulated_grads[i].add_(p.grad)

    model.zero_grad(set_to_none=True)
    for p, grad in zip(trainable_params, accumulated_grads):
        p.grad = grad.div_(total_microsteps) if grad is not None else None
    return train_loss, x, y, epoch
