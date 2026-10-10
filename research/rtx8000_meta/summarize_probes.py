"""Print cross-batch probe results. Negative delta-B means B loss decreased."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+', type=Path)
    args = parser.parse_args()
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
