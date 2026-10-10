"""Print cross-batch probe results. Negative delta-B means B loss decreased."""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


ALIGNMENT_FIELDS = ('cosine_a_b', 'b_loss_change', 'cosine_a_b_prime',
                    'relative_b_gradient_change', 'cosine_a_b_correction',
                    'cosine_b_b_prime', 'b_prime_norm_ratio')


def summarize_alignment(files, aggregate=False):
    print('file / update / pair / group / radius | cos(A,B) | delta-B | cos(A,Bprime) | relative-dBgrad | cos(A,correction) | cos(B,Bprime) | norm(Bprime)/norm(B)')
    grouped = defaultdict(list)
    def fmt(v): return 'n/a' if v is None else f'{v:+.6f}'
    for path in files:
        with path.open() as stream:
            for line in stream:
                record = json.loads(line)
                for m in record['measurements']:
                    for t in m['trials']:
                        if t['kind'] != 'raw_a' or 'cosine_a_b_prime' not in t: continue
                        values = [m['cosine_a_b']] + [t[key] for key in ALIGNMENT_FIELDS[1:]]
                        key = (str(path), record['update'], m['mode'], t['radius'])
                        grouped[key].append(values)
                        if not aggregate:
                            print(f"{path.parent.name} / {record['update']} / {m.get('pair_index', 0)} / {m['mode']} / {t['radius']} | " + ' | '.join(fmt(v) for v in values))
    if aggregate:
        for (path, update, mode, radius), rows in grouped.items():
            summaries = []
            for column in zip(*rows):
                valid = [v for v in column if v is not None]
                if not valid: summaries.append('n/a (0 defined)'); continue
                std = fmt(statistics.stdev(valid)) if len(valid) > 1 else 'n/a'
                summaries.append(f'{fmt(statistics.mean(valid))} ± {std} (n={len(valid)})')
            hurt = sum(row[1] > 0 for row in rows)
            print(f'{Path(path).parent.name} / {update} / {len(rows)} pairs / {mode} / {radius} | ' + ' | '.join(summaries) + f' | B-hurt={hurt}/{len(rows)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+', type=Path)
    parser.add_argument('--alignment', action='store_true', help='Show adapted-B gradient measurements')
    parser.add_argument('--aggregate', action='store_true', help='Mean/std across pairs, grouped by file, update, mode and radius')
    args = parser.parse_args()
    if args.aggregate and not args.alignment: parser.error('--aggregate requires --alignment')
    if args.alignment:
        summarize_alignment(args.files, args.aggregate)
        return
    print('file / update / group | weight-norm | cos(A,B) | raw@0.5 dB | Muon@0.5 dB | native-Muon dB | decay dB | combined dB')
    def number(value): return 'n/a' if value is None else f'{value:+.6f}'
    for path in args.files:
        with path.open() as stream:
            for line in stream:
                record = json.loads(line)
                for m in record['measurements']:
                    def delta(kind, radius=None):
                        return next((t['b_loss_change'] for t in m['trials'] if t['kind'] == kind and t['radius'] == radius), None)
                    values = [m['weight_norm'], m['cosine_a_b'], delta('raw_a', .5),
                              delta('muon_a_matched_radius', .5), delta('muon_a_native'),
                              delta('decay_a_native'), delta('muon_a_plus_decay_native')]
                    print(f"{path.parent.name} / {record['update']} / {m['mode']} | " + ' | '.join(number(v) for v in values))


if __name__ == '__main__': main()
