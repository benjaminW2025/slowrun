"""Read-only cross-batch probes; no optimizer step or training gradients are mutated."""
from contextlib import contextmanager
import math

import torch
from torch import nn

from virtual_rank import select_trial_parameters


def _norm(xs):
    return float(torch.stack([x.float().square().sum() for x in xs]).sum().sqrt())


def _dot(xs, ys):
    return float(torch.stack([(x.float() * y.float()).sum() for x, y in zip(xs, ys)]).sum())


def _cos(xs, ys):
    a, b = _norm(xs), _norm(ys)
    return _dot(xs, ys) / (a * b) if a > 0 and b > 0 else None


def _ratio(a, b):
    return a / b if b > 0 else None


def _interaction(ga, gb, gb_prime):
    correction = [new - old for new, old in zip(gb_prime, gb)]
    bnorm, pnorm, cnorm = _norm(gb), _norm(gb_prime), _norm(correction)
    return dict(cosine_a_b_prime=_cos(ga, gb_prime),
                relative_b_gradient_change=_ratio(cnorm, bnorm),
                cosine_a_b_correction=_cos(ga, correction),
                cosine_b_b_prime=_cos(gb, gb_prime),
                b_prime_norm=pnorm, b_prime_norm_ratio=_ratio(pnorm, bnorm),
                b_correction_norm=cnorm)


@contextmanager
def _probe_context(model, params):
    originals = [p.detach().clone() for p in params]
    modes = [(m, m.training) for m in model.modules()]
    # Keep the training recurrence/backprop policy, but remove dropout noise.
    # fork_rng ensures even an unexpected stochastic forward cannot alter training RNG.
    devices = sorted({p.device.index for p in model.parameters() if p.device.type == 'cuda'})
    with torch.random.fork_rng(devices=devices):
        try:
            for m in model.modules():
                if isinstance(m, nn.Dropout): m.training = False
            yield originals
        finally:
            with torch.no_grad(): torch._foreach_copy_(params, originals)
            for m, training in modes: m.training = training


def _gradients(forward_loss, batches, params, scale):
    total = [torch.zeros_like(p) for p in params]
    loss_sum = 0.0
    for x, y in batches:
        loss = forward_loss(x, y)
        loss_sum += float(loss.detach())
        # Avoid the same FP16 loss-predivision underflow as the training path.
        grads = torch.autograd.grad(loss * scale, params, allow_unused=True)
        if any(grad is None for grad in grads):
            raise RuntimeError(
                "Selected probe matrices are disconnected from the loss graph. "
                "Missing gradients are not zero gradients; check autocast caching "
                "across no-grad recurrent passes."
            )
        for out, grad in zip(total, grads):
            out.add_(grad.detach())
    for out in total: out.div_(len(batches) * scale)
    if not all(bool(torch.isfinite(g).all()) for g in total):
        raise FloatingPointError('Nonfinite diagnostic gradient')
    return loss_sum / len(batches), total


@torch.no_grad()
def preview_muon(optimizer, params, grads, muon_step):
    """Run the production Muon kernel on private copies of selected state slices.

    Returns native displacement without decay and native configured-decay displacement.
    World-size-one optimizer state layout is required, as in this experiment.
    """
    ids = {id(p): i for i, p in enumerate(params)}
    descent, decay, masks = [None] * len(params), [None] * len(params), [None] * len(params)
    for group in optimizer.param_groups:
        if group['kind'] != 'muon': continue
        indices = [i for i, p in enumerate(group['params']) if id(p) in ids]
        if not indices: continue
        selected = [group['params'][i] for i in indices]
        slots = [ids[id(p)] for p in selected]
        stacked = torch.stack([grads[i] for i in slots])
        # Zero-initialized scratch parameters directly reveal -lr * transformed_gradient.
        scratch = torch.zeros_like(stacked)
        state = optimizer.state.get(group['params'][0], {})
        rows, cols = selected[0].shape
        def buffer(key, shape):
            value = state.get(key)
            return value[indices].clone() if value is not None else stacked.new_zeros(shape)
        momentum = buffer('momentum_buffer', stacked.shape)
        shape = (len(indices), rows, 1) if rows >= cols else (len(indices), 1, cols)
        second = buffer('second_momentum_buffer', shape)
        lr = group['lr'] * max(1., rows / cols) ** .5
        # Production uses scalar FP32 tensors; use the same types here.
        scalar = lambda x: torch.tensor(x, dtype=torch.float32)
        muon_step(stacked, scratch, momentum, second, scalar(group['momentum']), scalar(lr),
                  scalar(0.), scalar(group['beta2']),
                  torch.ones((len(indices), 1, 1), dtype=torch.bool, device=stacked.device),
                  group['ns_steps'], -1 if rows >= cols else -2)
        for slot, p, delta in zip(slots, selected, scratch.unbind()):
            descent[slot] = delta.clone()
            # d = -lr*g, so g*w >= 0 iff d*w <= 0 (all eligible LRs are positive).
            mask = ((delta * p) <= 0 if group.get("weight_decay_mode", "cautious") == "cautious"
                    else torch.ones_like(p, dtype=torch.bool))
            masks[slot] = mask
            decay[slot] = -lr * group['weight_decay'] * p.detach() * mask
    if any(x is None for x in descent): raise ValueError('Probe includes non-Muon parameters')
    if not all(bool(torch.isfinite(d).all()) for d in descent + decay):
        raise FloatingPointError('Nonfinite Muon preview')
    return descent, decay, masks


def probe_episode(model, optimizer, named_parameters, a_batches, b_batches, forward_loss,
                  muon_step, scale, modes=('mlp',), radii=(.1, .5), alignment_only=False):
    """Probe the first virtual-rank episode, at unadapted theta before the real update.

    Existing p.grad contains the actual averaged meta-training gradient, still scaled.
    autograd.grad leaves it untouched. Preview state is cloned, never optimizer.step().
    """
    if not a_batches or not b_batches: raise ValueError('Probe requires both A and B')
    if scale <= 0 or not math.isfinite(scale): raise ValueError('Invalid gradient scale')
    if any(r <= 0 or not math.isfinite(r) for r in radii): raise ValueError('Invalid probe radius')
    named_parameters = list(named_parameters)
    muon_ids = {id(p) for g in optimizer.param_groups if g['kind'] == 'muon' for p in g['params']}
    names = {id(p): n for n, p in named_parameters}
    results = []
    for mode in modes:
        params = select_trial_parameters(named_parameters, muon_ids, mode)
        if not params: raise ValueError('Probe mode must select matrices')
        with _probe_context(model, params) as originals:
            loss_a, ga = _gradients(forward_loss, a_batches, params, scale)
            loss_b, gb = _gradients(forward_loss, b_batches, params, scale)
            # Missing training gradients must not silently become active zero updates.
            if not alignment_only:
                if any(p.grad is None for p in params): raise ValueError('Missing selected training gradient')
                train_grads = [p.grad.detach() / scale for p in params]
                da, wa, ma = preview_muon(optimizer, params, ga, muon_step)
                dt, wt, mt = preview_muon(optimizer, params, train_grads, muon_step)
            weight_norm, anorm, bnorm = _norm(originals), _norm(ga), _norm(gb)
            muon_norm = _norm(da) if not alignment_only else 0.0
            tensors = []
            for i, (p, w, a, b) in enumerate(zip(params, originals, ga, gb)):
                wn = _norm([w])
                tensor = dict(name=names[id(p)], parameters=p.numel(), weight_norm=wn,
                              gradient_a_norm=_norm([a]), gradient_b_norm=_norm([b]),
                              cosine_a_b=_cos([a], [b]))
                if not alignment_only:
                    d, decay, mask = da[i], wa[i], ma[i]
                    dn = _norm([d])
                    tensor.update(cosine_b_muon_displacement=_cos([b], [d]),
                                  muon_a_step_norm=dn, muon_a_relative_step=_ratio(dn, wn),
                                  decay_a_norm=_norm([decay]), decay_a_fraction=float(mask.float().mean()))
                tensors.append(tensor)
            # Parameter-wise diagnostics above also let us aggregate by layer offline.
            def scenarios():
                # Construct one trial at a time: retaining every MLP-sized delta
                # simultaneously can consume several extra GB on a 16-GB card.
                for radius in radii:
                    yield 'raw_a', radius, [-radius * g / anorm if anorm > 0 else torch.zeros_like(g) for g in ga]
                if alignment_only: return
                if muon_norm > 0:
                    for radius in radii: yield 'muon_a_matched_radius', radius, [radius * d / muon_norm for d in da]
                yield 'muon_a_native', None, da
                yield 'decay_a_native', None, wa
                yield 'muon_a_plus_decay_native', None, [d + w for d, w in zip(da, wa)]
                yield 'muon_train_native', None, dt
                yield 'decay_train_native', None, wt
                yield 'muon_train_plus_decay_native', None, [d + w for d, w in zip(dt, wt)]
            trials = []
            for kind, radius, delta in scenarios():
                with torch.no_grad():
                    # Absolute reconstruction avoids add/undo floating-point drift.
                    for p, original, d in zip(params, originals, delta): p.copy_(original + d)
                if kind == 'raw_a':
                    trial_loss, gb_prime = _gradients(forward_loss, b_batches, params, scale)
                    interaction = _interaction(ga, gb, gb_prime)
                    interaction['matrices'] = [dict(name=names[id(p)], **_interaction([a], [b], [new]))
                                               for p, a, b, new in zip(params, ga, gb, gb_prime)]
                    del gb_prime
                else:
                    with torch.no_grad():
                        trial_loss = sum(float(forward_loss(x, y)) for x, y in b_batches) / len(b_batches)
                    interaction = {}
                dn = _norm(delta)
                predicted = _dot(gb, delta)
                trials.append(dict(kind=kind, radius=radius, step_norm=dn,
                                   relative_step=_ratio(dn, weight_norm), b_loss=trial_loss,
                                   b_loss_change=trial_loss - loss_b,
                                   first_order_b_loss_change=predicted,
                                   nonlinear_residual=trial_loss - loss_b - predicted, **interaction))
            results.append(dict(mode=mode, a_loss=loss_a, b_loss=loss_b, weight_norm=weight_norm,
                                gradient_a_norm=anorm, gradient_b_norm=bnorm, cosine_a_b=_cos(ga, gb),
                                cosine_b_muon_displacement=_cos(gb, da) if not alignment_only else None,
                                alignment_only=alignment_only, matrices=tensors, trials=trials))
    return results
