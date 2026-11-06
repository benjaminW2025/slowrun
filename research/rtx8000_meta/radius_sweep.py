"""Read-only, paired downhill-radius measurements. No optimizer steps."""
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from diagnostics import _probe_context, _gradients, _norm, _interaction, _cos


def batch_hash(batches):
    digest = hashlib.sha256()
    for x, y in batches:
        for tensor in (x, y):
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def prediction_kl(reference_logits, adapted_logits, targets, chunk=128):
    """Token-mean KL(p_original || p_adapted), in nats, over valid targets.

    Chunk the vocabulary softmax work by token; don't retain probabilities
    for every token and radius. Input logits may be resident on the GPU.
    """
    before = reference_logits.reshape(-1, reference_logits.shape[-1])
    after = adapted_logits.reshape_as(before)
    mask = targets.reshape(-1) != -1
    total, count = 0.0, 0
    for start in range(0, len(before), chunk):
        valid = mask[start:start + chunk]
        if not bool(valid.any()):
            continue
        p = before[start:start + chunk][valid].float().log_softmax(-1)
        q = after[start:start + chunk][valid].float().log_softmax(-1)
        value = (p.exp() * (p - q)).sum(-1)
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError('Nonfinite prediction KL')
        total += float(value.double().sum())
        count += len(value)
    if count == 0:
        raise ValueError('KL measurement has no valid targets')
    # Small negative values from floating-point cancellation are retained.
    return total / count, count


def measure_pair(model, params, a_batches, b_batches, forward_loss, forward_logits,
                 radii, scale=1.0):
    if not params or not a_batches or not b_batches:
        raise ValueError('Select parameters and nonempty A/B batches')
    if not radii or any(r < 0 or not math.isfinite(r) for r in radii):
        raise ValueError('Radii must be finite and nonnegative')
    # _gradients averages microbatch means; require equally sized, unmasked batches.
    sizes = {y.numel() for _, y in a_batches + b_batches}
    if len(sizes) != 1 or any(bool((y == -1).any()) for _, y in a_batches + b_batches):
        raise ValueError('Sweep requires equally sized, unmasked training microbatches')
    with _probe_context(model, params) as originals:
        base_a, ga = _gradients(forward_loss, a_batches, params, scale)
        base_b, gb = _gradients(forward_loss, b_batches, params, scale)
        anorm = _norm(ga)
        weight_norm = _norm(originals)

        @torch.no_grad()
        def restore():
            torch._foreach_copy_(params, originals)

        @torch.no_grad()
        def perturb(radius):
            restore()
            if anorm > 0 and radius > 0:
                torch._foreach_add_(params, ga, alpha=-radius / anorm)

        records = []
        for radius in radii:
            perturb(radius)
            if radius == 0 or anorm == 0:
                a_loss, b_loss, gb_prime, kl = base_a, base_b, gb, 0.0
            else:
                with torch.no_grad():
                    a_loss = sum(float(forward_loss(x, y)) for x, y in a_batches) / len(a_batches)
                b_loss, gb_prime = _gradients(forward_loss, b_batches, params, scale)
                total_kl, tokens = 0.0, 0
                # Recompute reference logits per radius to avoid multi-GB CPU caches.
                # At most one microbatch's original/adapted logits are live at once.
                with torch.no_grad():
                    for x, y in b_batches:
                        restore()
                        before = forward_logits(x).detach()
                        perturb(radius)
                        after = forward_logits(x).detach()
                        value, count = prediction_kl(before, after, y)
                        total_kl += value * count
                        tokens += count
                        del before, after
                kl = total_kl / tokens
            if not all(math.isfinite(v) for v in (a_loss, b_loss, kl)):
                raise FloatingPointError(f'Nonfinite sweep metric at radius {radius}')
            record = dict(radius=radius, a_loss=base_a, b_loss=base_b,
                          adapted_a_loss=a_loss, adapted_b_loss=b_loss,
                          a_loss_change=a_loss-base_a, b_loss_change=b_loss-base_b,
                          b_prediction_kl=kl, gradient_a_norm=anorm,
                          gradient_b_norm=_norm(gb), cosine_a_b=_cos(ga, gb),
                          weight_norm=weight_norm, zero_a_gradient=anorm == 0,
                          gradient_scope='selected_mlp_matrices')
            record.update(_interaction(ga, gb, gb_prime))
            records.append(record)
        return records


def run_sweep(model, params, batches, forward_loss, forward_logits, output,
              pairs, offset, microsteps_per_half, radii, scale, metadata):
    if pairs < 1 or offset < 0:
        raise ValueError('pairs must be positive and offset nonnegative')
    output = Path(output)
    destination = output / 'radius_pairs.jsonl'
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite {destination}')
    needed = (offset + pairs) * 2 * microsteps_per_half
    # Caller ensures this fits within a single traversal of the training corpus.
    metadata = dict(metadata, pairs=pairs, offset=offset, radii=radii,
                    microsteps_per_half=microsteps_per_half, batches_required=needed,
                    kl_direction='original || adapted', kl_units='nats per target token',
                    dropout=False, gradient_scope='selected_mlp_matrices')
    (output / 'radius_manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    for _ in range(offset * 2 * microsteps_per_half):
        next(batches)
    start = time.monotonic()
    with destination.open('x') as file:
        for index in range(offset, offset + pairs):
            halves = []
            for _ in range(2):
                halves.append([next(batches)[:2] for _ in range(microsteps_per_half)])
            a, b = halves
            tick = time.monotonic()
            rows = measure_pair(model, params, a, b, forward_loss, forward_logits, radii, scale)
            pair_seconds = time.monotonic() - tick
            hashes = dict(a_sha256=batch_hash(a), b_sha256=batch_hash(b))
            for row in rows:
                row.update(pair=index, pair_seconds=pair_seconds, **hashes)
                file.write(json.dumps(row, allow_nan=False) + '\n')
            file.flush()
            completed = index - offset + 1
            elapsed = time.monotonic() - start
            print(f'Radius sweep pair {completed}/{pairs}: {pair_seconds:.2f}s; '
                  f'estimated remaining {(pairs-completed)*elapsed/completed/60:.1f} min; '
                  f'100-pair measurement estimate {100*elapsed/completed/60:.1f} min', flush=True)
    print(f'Radius results: {destination}', flush=True)
