#!/usr/bin/env python3
"""Condition 3: does prompting with the SAME task beat prompting with a DIFFERENT one?

    PYTHONPATH=third_party/icrt:. python -m aiworker_icrt.condition3 \
        --dataset /dev/shm/icrt_multitask \
        --checkpoint runs/mt21_gpu/checkpoint-15.pth \
        --train-yaml runs/mt21_gpu/run.yaml \
        --split-dir runs/mt21_gpu --out runs/mt21_gpu/cond3

Conditions 1 and 2 -- prompting with the same task, scoring accuracy -- cannot
tell in-context learning apart from ordinary behaviour cloning. A policy that
ignores the prompt entirely scores identically on both. Condition 3 is the one
that separates them: hold the observations fixed and change only the task the
prompt comes from. If the model is reading the demonstration, that must hurt.

THREE THINGS ARE MEASURED, and the third is why this is trustworthy:

1. same-task vs cross-task MAE, prompt length held constant.
2. A SELF-PROMPT control: prompt with the exact episode about to be replayed.
   That is the strongest prompt possible -- the model has seen the answer. If it
   does not beat a cross-task prompt, no prompt helps.
3. Whether the predicted ACTIONS differ at all between prompt conditions.
   Equal MAE under different prompts has two explanations: the model ignores the
   prompt, or the prompt never arrives. Comparing raw predictions separates a
   real negative result from a plumbing bug -- and on this project that
   distinction has been wrong eight times.

PROMPT LENGTH IS A CONFOUND AND IS CONTROLLED. Our tasks have very different
episode lengths (ylw2 ~958 frames, box/ecu ~330), so an uncontrolled comparison
varies prompt length at the same time as prompt task. Every prompt is truncated
to its last --prompt-steps frames.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

from aiworker_icrt.evaluate import evaluate

# Which arm actually does the work, per task. Scoring the idle arm measures
# nothing: in ylw2 the left arm is frozen at exactly 0.000 m, so its error is
# trivially small and would dilute the comparison.
WORKING_ARM = {'ylw2': 'right', 'box': 'left', 'ecu': 'left'}


def _short(task: str) -> str:
    t = task.lower()
    return 'ylw2' if 'yellow' in t else 'ecu' if 'ecu' in t else 'box'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--train-yaml', type=Path, required=True)
    ap.add_argument('--split-dir', type=Path, required=True,
                    help='run directory holding train_split.json / val_split.json')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--steps', type=int, default=40)
    ap.add_argument('--prompt-steps', type=int, default=300)
    ap.add_argument('--prompts-per-task', type=int, default=2)
    a = ap.parse_args()

    verb = json.loads((a.dataset / 'verb_to_episode.json').read_text())
    train = set(json.loads((a.split_dir / 'train_split.json').read_text()))
    val = sorted(json.loads((a.split_dir / 'val_split.json').read_text()))
    ep2task = {e: _short(t) for t, eps in verb.items() for e in eps}

    # Prompts come ONLY from the train split, or the comparison leaks.
    prompts: dict[str, list[str]] = {}
    for task, eps in verb.items():
        s = _short(task)
        prompts[s] = sorted(e for e in eps if e in train and e.startswith(s))[:a.prompts_per_task]
    print('prompt episodes:', json.dumps(prompts, indent=1))

    rows = []
    for ev in val:
        et = ep2task[ev]
        arm = WORKING_ARM[et]
        for pt, peps in sorted(prompts.items()):
            for pe in peps:
                r = evaluate(a.dataset, a.checkpoint, a.train_yaml, pe, ev, a.out,
                             max_steps=a.steps, prompt_max_steps=a.prompt_steps)
                rows.append({'eval': ev, 'eval_task': et, 'prompt': pe,
                             'prompt_task': pt, 'same': pt == et, 'arm': arm,
                             'mae': r[arm]['pos_mae_m']})
                print('  %-22s eval=%-5s prompt=%-5s %-5s mae=%.4f'
                      % (ev, et, pt, 'SAME' if pt == et else 'CROSS', rows[-1]['mae']))
        # control: prompt with the episode itself
        r = evaluate(a.dataset, a.checkpoint, a.train_yaml, ev, ev, a.out,
                     max_steps=a.steps, prompt_max_steps=a.prompt_steps)
        rows.append({'eval': ev, 'eval_task': et, 'prompt': ev, 'prompt_task': et,
                     'same': None, 'arm': arm, 'mae': r[arm]['pos_mae_m']})
        print('  %-22s eval=%-5s prompt=SELF  mae=%.4f' % (ev, et, rows[-1]['mae']))

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'condition3.json').write_text(json.dumps(rows, indent=2))

    by = defaultdict(dict)
    for ev in val:
        rs = [r for r in rows if r['eval'] == ev]
        by[ev]['same'] = st.mean([r['mae'] for r in rs if r['same'] is True])
        by[ev]['cross'] = st.mean([r['mae'] for r in rs if r['same'] is False])
        by[ev]['self'] = [r['mae'] for r in rs if r['same'] is None][0]

    print('\n================ CONDITION 3 ================')
    print('%-22s %-5s %-9s %-9s %-9s %s' % ('eval', 'task', 'SELF', 'same', 'cross', 'same-cross'))
    for ev in val:
        d = by[ev]
        print('%-22s %-5s %-9.4f %-9.4f %-9.4f %+.4f'
              % (ev, ep2task[ev], d['self'], d['same'], d['cross'], d['same'] - d['cross']))
    ms = st.mean(d['same'] for d in by.values())
    mc = st.mean(d['cross'] for d in by.values())
    print('-' * 70)
    print('%-22s %-5s %-9.4f %-9.4f %-9.4f %+.4f'
          % ('MEAN', '', st.mean(d['self'] for d in by.values()), ms, mc, ms - mc))

    # Does the prompt reach the model at all? Equal MAE is ambiguous; equal
    # PREDICTIONS are not.
    print('\n=== is the prompt reaching the model? ===')
    for ev in val[:1]:
        et = ep2task[ev]
        other = next(t for t in prompts if t != et)
        fs = {k: sorted(a.out.glob(f'{ev}__prompt_{p}.npz'))
              for k, p in (('same', prompts[et][0]), ('cross', prompts[other][0]))}
        if all(fs.values()):
            x = np.load(fs['same'][0])['pred']
            y = np.load(fs['cross'][0])['pred']
            print('  %s: identical=%s  max|diff|=%.3e  mean|diff|=%.3e'
                  % (ev, np.array_equal(x, y), np.abs(x - y).max(), np.abs(x - y).mean()))
            print('  identical predictions would mean the prompt never arrives '
                  '(a bug); differing predictions with equal error mean the '
                  'model reads the prompt and learns nothing from it.')


if __name__ == '__main__':
    main()
