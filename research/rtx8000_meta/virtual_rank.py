"""Serial data-parallel A/B episodes for the RTX 8000 Slowrun experiment.

The returned parameter gradients equal the mean of independent virtual-rank
gradients. No optimizer step occurs until every trial has been restored.
"""

import math

import torch


TRIAL_PATTERNS = {
    "off": (),
    "mlp": (".mlp.",),
    "mlp_v": (".mlp.", ".attn.c_v."),
    "qk": (".attn.c_q.", ".attn.c_k."),
    "v": (".attn.c_v.",),
    "qkv": (".attn.c_q.", ".attn.c_k.", ".attn.c_v."),
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


def validate_gradients(named_parameters, expected_parameter_ids):
    """Reject disconnected/nonfinite optimizer gradients before any update.

    A tensor of zeros is allowed (e.g. QKV at initialization). None is not:
    every optimizer-managed parameter in this model participates in the loss.
    Parameters deliberately excluded from the optimizer, such as disabled XSA,
    are not checked.
    """
    expected = [(name, p) for name, p in named_parameters if id(p) in expected_parameter_ids]
    missing = [name for name, p in expected if p.grad is None]
    if missing:
        raise RuntimeError("Missing training gradients (disconnected parameters): " + ", ".join(missing))
    if not expected:
        raise ValueError("No optimizer parameters to validate")
    norms = torch.stack(torch._foreach_norm([p.grad for name, p in expected]))
    valid = torch.isfinite(norms)
    if not bool(valid.all()):
        bad = [name for (name, p), ok in zip(expected, valid.tolist()) if not ok]
        raise FloatingPointError("Nonfinite training gradients: " + ", ".join(bad))
    return len(expected)


@torch.no_grad()
def _temporary_step(trial_params, step_norm):
    original_weights = [p.detach().clone() for p in trial_params]
    if any(p.grad is None for p in trial_params):
        raise RuntimeError("Missing temporary A gradients: selected matrices are disconnected from the loss")
    direction = [p.grad.detach().clone() for p in trial_params]
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


@torch.no_grad()
def _temporary_group_step(trial_params, groups):
    """Apply independent group norms atomically; zero-radius groups are untouched."""
    grouped = [p for params, radius in groups for p in params]
    if len({id(p) for p in grouped}) != len(grouped):
        raise ValueError("Temporary-step groups overlap")
    if {id(p) for p in grouped} != {id(p) for p in trial_params}:
        raise ValueError("Temporary-step groups must cover exactly the trial parameters")
    if any(not math.isfinite(radius) or radius < 0 for params, radius in groups):
        raise ValueError("Group step norms must be nonnegative and finite")
    original = [p.detach().clone() for p in trial_params]
    try:
        for params, radius in groups:
            if radius > 0:
                _temporary_step(params, radius)
    except BaseException:
        torch._foreach_copy_(trial_params, original)
        raise
    return original


def _collect_symmetric_gradients(model, trainable_params, trial_params, x, y, next_batch,
                                forward_loss, scaler, virtual_ranks, microsteps_per_half,
                                step_norm, capture_episode, capture_episodes, trial_groups):
    """Average A, B, B-at-A and A-at-B; replay each batch's dropout masks."""
    total = 4 * virtual_ranks * microsteps_per_half
    accumulated = [None] * len(trainable_params)
    loss_sum = x.new_zeros((), dtype=torch.float32)
    epoch = None
    devices = [x.device.index if x.device.index is not None else torch.cuda.current_device()] if x.is_cuda else []

    def add_gradients():
        for i, p in enumerate(trainable_params):
            if p.grad is not None:
                if accumulated[i] is None:
                    accumulated[i] = p.grad.detach().clone()
                else:
                    accumulated[i].add_(p.grad)

    for rank in range(virtual_ranks):
        batches, directions = [[], []], []
        destination = (capture_episodes[rank] if capture_episodes is not None
                       and rank < len(capture_episodes) else
                       capture_episode if rank == 0 else None)
        # Original gradients for both halves, before any temporary perturbation.
        for half in range(2):
            model.zero_grad(set_to_none=True)
            for _ in range(microsteps_per_half):
                cpu_rng = torch.get_rng_state()
                gpu_rng = torch.cuda.get_rng_state(devices[0]) if devices else None
                batches[half].append((x.detach(), y.detach(), cpu_rng, gpu_rng))
                if destination is not None:
                    destination[half].append((x.detach(), y.detach()))
                loss = forward_loss(x, y)
                loss_sum = loss_sum + loss.detach() / total
                scaler.scale(loss).backward()
                x, y, epoch = next_batch()
            if any(p.grad is None for p in trial_params):
                raise RuntimeError("Missing symmetric temporary-step gradients")
            directions.append([p.grad.detach().clone() for p in trial_params])
            add_gradients()
        for source, target in ((0, 1), (1, 0)):
            model.zero_grad(set_to_none=True)
            for p, grad in zip(trial_params, directions[source]):
                p.grad = grad
            original = None
            try:
                if trial_params:
                    original = (_temporary_group_step(trial_params, trial_groups)
                                if trial_groups is not None else
                                _temporary_step(trial_params, step_norm))
                model.zero_grad(set_to_none=True)
                # Reuse masks from the original evaluation, without advancing
                # the persistent RNG stream or consuming more training examples.
                with torch.random.fork_rng(devices=devices):
                    for bx, by, cpu_rng, gpu_rng in batches[target]:
                        torch.set_rng_state(cpu_rng)
                        if devices:
                            torch.cuda.set_rng_state(gpu_rng, devices[0])
                        loss = forward_loss(bx, by)
                        loss_sum = loss_sum + loss.detach() / total
                        scaler.scale(loss).backward()
                add_gradients()
            finally:
                if original is not None:
                    with torch.no_grad():
                        torch._foreach_copy_(trial_params, original)
    model.zero_grad(set_to_none=True)
    for p, grad in zip(trainable_params, accumulated):
        p.grad = grad.div_(total) if grad is not None else None
    return loss_sum, x, y, epoch


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
    capture_episodes=None,
    trial_groups=None,
    symmetric=False,
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

    if capture_episode is not None and capture_episodes is not None:
        raise ValueError("Use capture_episode or capture_episodes, not both")
    if symmetric:
        return _collect_symmetric_gradients(
            model, trainable_params, trial_params, x, y, next_batch, forward_loss, scaler,
            virtual_ranks, microsteps_per_half, step_norm, capture_episode, capture_episodes,
            trial_groups)
    for virtual_rank in range(virtual_ranks):
        model.zero_grad(set_to_none=True)
        original_weights = None
        try:
            for half in range(2):
                for _ in range(microsteps_per_half):
                    destination = (capture_episodes[virtual_rank] if capture_episodes is not None
                                   and virtual_rank < len(capture_episodes) else
                                   capture_episode if virtual_rank == 0 else None)
                    if destination is not None:
                        destination[half].append((x.detach(), y.detach()))
                    loss = forward_loss(x, y)
                    train_loss = train_loss + loss.detach() / total_microsteps
                    scaler.scale(loss).backward()
                    x, y, epoch = next_batch()
                if half == 0 and trial_params:
                    original_weights = (_temporary_group_step(trial_params, trial_groups)
                                        if trial_groups is not None else
                                        _temporary_step(trial_params, step_norm))
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
