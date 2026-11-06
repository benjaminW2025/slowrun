"""Summarize completed pairs and write dependency-free HTML scatter plots."""
import argparse
import csv
import html
import json
from pathlib import Path
import statistics


def percentile(values, q):
    values = sorted(values)
    at = (len(values)-1)*q
    lo = int(at)
    return values[lo] + (values[min(lo+1, len(values)-1)]-values[lo])*(at-lo)


def summarize(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows: raise ValueError('No completed measurements')
    groups = {}
    for row in rows: groups.setdefault(row['radius'], []).append(row)
    # Reject incomplete/misaligned pair sets; avoid silently plotting biased aggregates.
    pair_sets = [{r['pair'] for r in group} for group in groups.values()]
    if any(s != pair_sets[0] for s in pair_sets):
        raise ValueError('Radius groups have different completed pair sets')
    summaries = []
    metrics = ('a_loss_change', 'b_loss_change', 'b_prediction_kl', 'relative_b_gradient_change')
    for radius, group in sorted(groups.items()):
        entry = dict(radius=radius, pairs=len(group), fraction_b_improved=sum(r['b_loss_change'] < 0 for r in group)/len(group))
        for key in metrics:
            values = [r[key] for r in group if r[key] is not None]
            entry[key+'_count'] = len(values)
            for label, fn in (('mean', statistics.mean), ('p10', lambda v: percentile(v,.1)),
                              ('median', statistics.median), ('p90', lambda v: percentile(v,.9))):
                entry[key+'_'+label] = fn(values) if values else None
        summaries.append(entry)
    with path.with_name('radius_summary.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)
    with path.with_name('radius_pairs.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    colors = ['#64748b', '#2563eb', '#0d9488', '#d97706', '#dc2626', '#9333ea', '#db2777']
    palette = {r: colors[i % len(colors)] for i, r in enumerate(sorted(groups))}
    def plot(xkey, ykey, title):
        points = [r for r in rows if r[xkey] is not None and r[ykey] is not None]
        if not points: return ''
        xs, ys = [r[xkey] for r in points], [r[ykey] for r in points]
        xmin,xmax=min(0,min(xs)),max(xs); ymin,ymax=min(0,min(ys)),max(0,max(ys))
        if xmax == xmin: xmax=xmin+1e-9
        if ymax == ymin: ymax=ymin+1e-9
        x=lambda v: 90+620*(v-xmin)/(xmax-xmin)
        y=lambda v: 310-250*(v-ymin)/(ymax-ymin)
        svg=[f'<h2>{html.escape(title)}</h2><svg viewBox="0 0 770 365" role="img" aria-label="{html.escape(title)}">',
             '<path d="M90 60V310H710" fill="none" stroke="#64748b"/>',
             f'<line x1="90" x2="710" y1="{y(0)}" y2="{y(0)}" stroke="#aaa" stroke-dasharray="4 4"/>']
        for i in range(5):
            xv=xmin+(xmax-xmin)*i/4; yv=ymin+(ymax-ymin)*i/4
            svg += [f'<text x="{x(xv)}" y="330" text-anchor="middle">{xv:.3g}</text>',
                    f'<text x="80" y="{y(yv)+4}" text-anchor="end">{yv:.3g}</text>']
        for r in points:
            label=f"pair {r['pair']}, radius {r['radius']}: {xkey}={r[xkey]:.6g}, {ykey}={r[ykey]:.6g}"
            svg.append(f'<circle cx="{x(r[xkey])}" cy="{y(r[ykey])}" r="3" fill="{palette[r["radius"]]}" opacity=".6"><title>{html.escape(label)}</title></circle>')
        svg += [f'<text x="400" y="355" text-anchor="middle">{xkey}</text>',
                f'<text transform="translate(18 190) rotate(-90)" text-anchor="middle">{ykey}</text></svg>']
        return ''.join(svg)
    legend=' '.join(f'<span style="color:{c}">● ρ={r:g}</span>' for r,c in palette.items())
    body='<h1>Checkpoint radius sweep</h1><p>Each dot is one pair at one radius. Hover for values. Negative B loss change means improvement. KL is original || adapted, nats per token.</p>'+legend
    for xkey,ykey,title in [('radius','b_loss_change','Transfer versus parameter radius'),
                            ('b_prediction_kl','b_loss_change','Transfer versus prediction change'),
                            ('radius','relative_b_gradient_change','B gradient sensitivity'),
                            ('radius','a_loss_change','Adaptation on A')]:
        body+=plot(xkey,ykey,title)
    body+='<h2>Per-radius summary</h2><table><tr><th>Radius</th><th>Pairs</th><th>Mean ΔB</th><th>10–90% ΔB</th><th>B improves</th><th>Mean KL</th></tr>'
    for s in summaries:
        body+=f'<tr><td>{s["radius"]:g}</td><td>{s["pairs"]}</td><td>{s["b_loss_change_mean"]:.6f}</td><td>{s["b_loss_change_p10"]:.6f} … {s["b_loss_change_p90"]:.6f}</td><td>{s["fraction_b_improved"]:.1%}</td><td>{s["b_prediction_kl_mean"]:.6g}</td></tr>'
    body+='</table><p>These are descriptive measurements, not independent hypothesis tests or evidence of held-out generalization. Gradient changes concern selected MLP matrices only. Pairs need not come from distinct documents.</p>'
    report=path.with_name('radius_report.html')
    report.write_text('<!doctype html><meta charset="utf-8"><title>Radius sweep</title><style>body{font:16px system-ui;max-width:900px;margin:40px auto;padding:0 20px;color:#172033}svg{width:100%;background:#f8fafc}svg text{font:12px system-ui}td,th{padding:8px;text-align:right}span{margin-right:15px}</style>'+body)
    print(f'Summary: {path.with_name("radius_summary.csv")}\nPlots: {report}')
    return summaries


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('jsonl', type=Path)
    summarize(parser.parse_args().jsonl)
