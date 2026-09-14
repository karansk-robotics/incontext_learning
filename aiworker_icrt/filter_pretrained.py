#!/usr/bin/env python3
"""Strip the tensors that cannot fit our retargeted model, so the rest will load.

    python -m aiworker_icrt.filter_pretrained \
        --src checkpoints_icrt/icrt_vitb_droid_pretrained/icrt_vitb_droid_pretrained.pth \
        --out checkpoints_icrt/icrt_trunk_21d.pth

`misc.load_model` calls `load_state_dict(..., strict=False)`. That tolerates
missing and unexpected keys but NOT shape mismatches -- those raise a
RuntimeError regardless of `strict`. 24 tensors mismatch, all of them at the
boundary of our retarget from DROID's 10-D single-arm / 2-camera space:

    icrt_proprio_encoder.fc1/fc2   (10,10)/(768,10) -> (21,21)/(768,21)
    icrt_action_encoder.fc1/fc2    (10,10)/(768,10) -> (21,21)/(768,21)
    icrt_action_decoder.mlp.fc2    (160,128)        -> (336,128)   16x10 -> 16x21
    icrt_attn_pooling.{0,1}.*      width 384        -> 256         2 cams -> 3
    icrt_attn_pooling.2.*          absent           -> our 3rd camera

Everything else -- the whole transformer trunk -- transfers.
"""

from __future__ import annotations

import argparse

import torch

def _sd(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    return ck['model'] if 'model' in ck else ck


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--reference', required=True,
                    help='a checkpoint of OUR model, used purely as a shape '
                         'reference -- must match the config you will train with '
                         '(proprio/action dim and num_cameras)')
    a = ap.parse_args()

    # Drop by MEASURED shape mismatch, not by name prefix. Prefix matching is
    # too blunt: icrt_attn_pooling holds `latent`, `q` and `kv` tensors that are
    # 768-wide and transfer fine, and icrt_proprio_encoder.fc2.bias is (768,) in
    # both. Dropping those whole prefixes threw away 3.6M parameters that fit
    # (111 tensors kept instead of 125).
    sd = _sd(a.src)
    ref = _sd(a.reference)

    kept, dropped = {}, []
    for k, v in sd.items():
        if k in ref and ref[k].shape == v.shape:
            kept[k] = v
        else:
            dropped.append(k)

    n_kept = sum(v.numel() for v in kept.values())
    n_all = sum(v.numel() for v in sd.values())
    print(f'kept {len(kept)} of {len(sd)} tensors, '
          f'{n_kept/1e6:.1f}M of {n_all/1e6:.1f}M params ({100*n_kept/n_all:.1f}%)')
    print(f'dropped {len(dropped)}:')
    for k in dropped:
        tgt = tuple(ref[k].shape) if k in ref else 'ABSENT in reference'
        print(f'   {k:<44} theirs {str(tuple(sd[k].shape)):<16} ours {tgt}')

    torch.save({'model': kept}, a.out)
    print(f'\nwrote {a.out}')
    print('Expect "kept 125 of 149 tensors, 88.6M". Anything else -- stop and '
          'find out why before training on it.')


if __name__ == '__main__':
    main()
