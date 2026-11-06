"""Launch a paired checkpoint radius sweep using the checkpoint's saved configuration."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--config', type=Path, help='Training result.json; defaults to checkpoint directory or parent')
    p.add_argument('--pairs', type=int, default=10)
    p.add_argument('--offset', type=int, default=0)
    p.add_argument('--radii', default='0,0.03,0.1,0.3,0.5,1.0')
    p.add_argument('--gpu', default='1')
    p.add_argument('--iterations', type=int, help='Defaults to trained maximum recurrence; set explicitly for an early checkpoint')
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    checkpoint = args.checkpoint.resolve(strict=True)
    config = args.config
    if config is None:
        config = next((folder/'result.json' for folder in (checkpoint.parent, checkpoint.parent.parent)
                       if (folder/'result.json').exists()), None)
    if config is None:
        p.error('Provide --config path/to/training/result.json for this checkpoint')
    config = config.resolve(strict=True)
    result = json.loads(config.read_text())
    saved = result['arguments']
    if args.pairs < 1 or args.offset < 0:
        p.error('pairs must be positive and offset nonnegative')
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    iterations = args.iterations or saved.get('num_iterations', 2)
    name = f'radius-sweep-{time.strftime("%Y%m%d-%H%M%S")}-{os.getpid()}'
    command = [sys.executable, str(here/'train.py'), '--init-model', str(checkpoint),
               '--radius-sweep-pairs', str(args.pairs), '--radius-sweep-offset', str(args.offset),
               '--radius-sweep-radii', args.radii, '--radius-sweep-config', str(config),
               '--mg-mode', 'off', '--no-compile', '--probe-every', '0',
               '--iteration-schedule', 'constant', '--num-iterations', str(iterations),
               '--schedule-steps', '500', '--max-train-steps', '1',
               '--eval-tokens', '1048576', '--run-name', name]
    keys = ('n_layer', 'n_embd', 'n_head', 'sequence_length', 'device_batch_size',
            'virtual_ranks', 'total_batch_size', 'precision', 'loss_scale', 'dropout', 'xsa_mode', 'seed')
    underscore_flags = {'n_layer', 'n_embd', 'n_head'}
    for key in keys:
        if key in saved:
            flag = key if key in underscore_flags else key.replace('_', '-')
            command.extend(['--'+flag, str(saved[key])])
    # Respect the source run's corpus paths (relative paths resolve in repo root).
    for key in ('input_bin', 'input_val_bin'):
        if saved.get(key): command.extend(['--'+key, str(saved[key])])
    if saved.get('no_doc_shuffle'): command.append('--no-doc-shuffle')
    print(shlex.join(command), flush=True)
    print(f'{args.pairs} pairs; {iterations} recurrent passes; dropout disabled; training backprop policy retained.', flush=True)
    if not args.execute:
        print('Dry run. Add --execute to measure; no training updates will occur.')
        return
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, WANDB_MODE='disabled')
    subprocess.run(command, cwd=root, env=env, check=True)
    subprocess.run([sys.executable, str(here/'summarize_radius_sweep.py'),
                    str(root/'runs'/name/'radius_pairs.jsonl')], check=True)


if __name__ == '__main__': main()
