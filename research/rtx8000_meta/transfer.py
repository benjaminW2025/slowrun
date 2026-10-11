"""Muon B-prime and held-out C probes. No weights, gradients or state are committed."""
import math
import torch

from diagnostics import (_norm, _dot, _cos, _ratio, _interaction,
                         _probe_context, _gradients, preview_muon)
from virtual_rank import select_trial_parameters


def agreement_component(ga, gb):
    """Coordinate-wise shared sign and minimum magnitude; a basis-dependent heuristic.

    paired = agreement + remainder exactly, up to floating-point rounding.
    This is NOT a unique or parameterization-invariant shared subspace.
    """
    return [torch.where((a > 0) & (b > 0), torch.minimum(a, b),
                        torch.where((a < 0) & (b < 0), torch.maximum(a, b),
                                    torch.zeros_like(a))) for a, b in zip(ga, gb)]


def _mean(ga, gb):
    return [(a + b) * .5 for a, b in zip(ga, gb)]


def _restore(params, originals, delta=None):
    with torch.no_grad():
        for i, (p, original) in enumerate(zip(params, originals)):
            p.copy_(original if delta is None else original + delta[i])


def _loss(forward_loss, batches):
    with torch.no_grad():
        value = sum(float(forward_loss(x, y)) for x, y in batches) / len(batches)
    if not math.isfinite(value): raise FloatingPointError('Nonfinite transfer loss')
    return value


def _muon(optimizer, params, grads, muon_step):
    # Every call clones the SAME optimizer state. No candidate updates history.
    direction, decay, masks = preview_muon(optimizer, params, grads, muon_step)
    return direction  # Exclude weight decay to isolate the gradient transformation.


def _muon_metrics(da, db, db_prime, paired_direction, meta_direction, ga):
    correction = [new - old for new, old in zip(db_prime, db)]
    raw_cos = _cos(ga, db_prime)
    return dict(cosine_muon_a_b_prime=_cos(da, db_prime),
                cosine_raw_a_muon_b_prime=None if raw_cos is None else -raw_cos,
                cosine_muon_b_b_prime=_cos(db, db_prime),
                cosine_muon_a_correction=_cos(da, correction),
                relative_muon_b_change=_ratio(_norm(correction), _norm(db)),
                muon_b_prime_step_norm=_norm(db_prime),
                muon_b_prime_norm_ratio=_ratio(_norm(db_prime), _norm(db)),
                cosine_muon_paired_meta=_cos(paired_direction, meta_direction),
                relative_muon_paired_meta_change=_ratio(
                    _norm([new - old for new, old in zip(meta_direction, paired_direction)]),
                    _norm(paired_direction)))


def _evaluate_c(source, gradients, model_params, originals, optimizer, muon_step,
                forward_loss, c_batches, gc, c_loss, radius, muon_direction=None):
    """C is used ONLY to score already-constructed directions, never to select them."""
    rows = []
    for transform in ('raw', 'muon'):
        native = ([-g for g in gradients] if transform == 'raw' else
                  muon_direction if muon_direction is not None else
                  _muon(optimizer, model_params, gradients, muon_step))
        native_norm = _norm(native)
        delta = [d * (radius / native_norm) if native_norm > 0 else torch.zeros_like(d) for d in native]
        if native_norm == 0:
            after = c_loss
        else:
            _restore(model_params, originals, delta)
            try:
                after = _loss(forward_loss, c_batches)
            finally:
                _restore(model_params, originals)
        predicted = _dot(gc, delta)
        rows.append(dict(source=source, transform=transform, radius=radius,
                         input_gradient_norm=_norm(gradients),
                         native_direction_norm=native_norm, applied_step_norm=_norm(delta),
                         native_linearized_c_loss_change=_dot(gc, native),
                         no_op=native_norm == 0, cosine_c_displacement=_cos(gc, delta),
                         c_loss=after, c_loss_change=after - c_loss,
                         linearized_c_loss_change=predicted,
                         nonlinear_residual=after - c_loss - predicted,
                         weight_decay_applied=False))
    return rows


def probe_transfer_episode(model, optimizer, named_parameters, a_batches, b_batches,
                           forward_loss, muon_step, scale, modes=('qk', 'v', 'mlp'),
                           radii=(.1, .5), c_batches=None, transfer_radius=.1):
    """Raw A->B-prime plus Muon comparisons, optionally scored on separate C.

    All trial displacements are applied at ORIGINAL theta. B-prime is computed
    at theta+d_A, then theta is restored BEFORE previewing the real update.
    Muon(A+B-prime) is computed directly, never by averaging Muon(A)/Muon(B-prime).
    """
    if not a_batches or not b_batches or c_batches == []:
        raise ValueError('A, B and requested C batches must be nonempty')
    if not math.isfinite(scale) or scale <= 0: raise ValueError('Invalid gradient scale')
    if not radii or any(not math.isfinite(r) or r <= 0 for r in radii):
        raise ValueError('Invalid adaptation radius')
    if not math.isfinite(transfer_radius) or transfer_radius <= 0:
        raise ValueError('Invalid transfer radius')
    named = list(named_parameters)
    muon_ids = {id(p) for g in optimizer.param_groups if g['kind'] == 'muon' for p in g['params']}
    names = {id(p): name for name, p in named}
    results = []
    for mode in modes:
        params = select_trial_parameters(named, muon_ids, mode)
        if not params: raise ValueError('Transfer probe must select matrices')
        with _probe_context(model, params) as originals:
            loss_a, ga = _gradients(forward_loss, a_batches, params, scale)
            loss_b, gb = _gradients(forward_loss, b_batches, params, scale)
            if not math.isfinite(loss_a) or not math.isfinite(loss_b):
                raise FloatingPointError('Nonfinite A/B probe loss')
            if c_batches is not None:
                loss_c, gc = _gradients(forward_loss, c_batches, params, scale)
                if not math.isfinite(loss_c): raise FloatingPointError('Nonfinite C probe loss')
            else:
                loss_c, gc = None, None
            da = _muon(optimizer, params, ga, muon_step)
            db = _muon(optimizer, params, gb, muon_step)
            paired = _mean(ga, gb)
            dpaired = _muon(optimizer, params, paired, muon_step)
            base = []
            if c_batches is not None:
                def candidates():
                    yield 'a', ga, da
                    yield 'b', gb, db
                    yield 'paired', paired, dpaired
                    yield 'zero_input', [torch.zeros_like(g) for g in ga], None
                    shared = agreement_component(ga, gb)
                    yield 'agreement', shared, None
                    yield 'remainder', [p - s for p, s in zip(paired, shared)], None
                for source, grads, direction in candidates():
                    base.extend(_evaluate_c(source, grads, params, originals, optimizer, muon_step,
                                            forward_loss, c_batches, gc, loss_c, transfer_radius, direction))
            trials = []
            anorm = _norm(ga)
            for radius in radii:
                delta_a = [-radius * g / anorm if anorm > 0 else torch.zeros_like(g) for g in ga]
                _restore(params, originals, delta_a)
                try:
                    after_b, gb_prime = _gradients(forward_loss, b_batches, params, scale)
                finally:
                    _restore(params, originals)
                if not math.isfinite(after_b): raise FloatingPointError('Nonfinite adapted B loss')
                db_prime = _muon(optimizer, params, gb_prime, muon_step)
                meta = _mean(ga, gb_prime)
                dmeta = _muon(optimizer, params, meta, muon_step)
                transfer = []
                if c_batches is not None:
                    transfer.extend(_evaluate_c('adapted_b', gb_prime, params, originals, optimizer, muon_step,
                                                forward_loss, c_batches, gc, loss_c, transfer_radius, db_prime))
                    transfer.extend(_evaluate_c('meta', meta, params, originals, optimizer, muon_step,
                                                forward_loss, c_batches, gc, loss_c, transfer_radius, dmeta))
                    transfer.extend(_evaluate_c('meta_correction', [(new - old) * .5 for new, old in zip(gb_prime, gb)],
                                                params, originals, optimizer, muon_step,
                                                forward_loss, c_batches, gc, loss_c, transfer_radius))
                predicted_b = _dot(gb, delta_a)
                muon_metrics = _muon_metrics(da, db, db_prime, dpaired, dmeta, ga)
                muon_metrics['matrices'] = [dict(name=names[id(p)], **_muon_metrics(
                    [a], [b], [bp], [paired_d], [meta_d], [raw_a]))
                    for p, a, b, bp, paired_d, meta_d, raw_a in zip(params, da, db, db_prime, dpaired, dmeta, ga)]
                interaction = _interaction(ga, gb, gb_prime)
                interaction['matrices'] = [dict(name=names[id(p)], **_interaction([a], [b], [bp]))
                                          for p, a, b, bp in zip(params, ga, gb, gb_prime)]
                trials.append(dict(kind='raw_a', radius=radius, step_norm=_norm(delta_a),
                                   b_loss=after_b, b_loss_change=after_b - loss_b,
                                   first_order_b_loss_change=predicted_b,
                                   nonlinear_residual=after_b - loss_b - predicted_b,
                                   **interaction,
                                   muon_b_prime=muon_metrics,
                                   transfer=transfer))
            results.append(dict(mode=mode, a_loss=loss_a, b_loss=loss_b, c_loss=loss_c,
                                weight_norm=_norm(originals), gradient_a_norm=_norm(ga),
                                gradient_b_norm=_norm(gb),
                                gradient_c_norm=None if gc is None else _norm(gc),
                                cosine_a_b=_cos(ga, gb),
                                cosine_a_c=None if gc is None else _cos(ga, gc),
                                cosine_b_c=None if gc is None else _cos(gb, gc),
                                cosine_muon_a_b=_cos(da, db),
                                agreement_definition='same_sign_min_magnitude',
                                matrices=[dict(name=names[id(p)], weight_norm=_norm([original]),
                                               gradient_a_norm=_norm([a]), gradient_b_norm=_norm([b]),
                                               cosine_a_b=_cos([a], [b]))
                                          for p, original, a, b in zip(params, originals, ga, gb)],
                                transfer_baselines=base, trials=trials))
    return results
