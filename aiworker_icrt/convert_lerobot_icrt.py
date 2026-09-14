#!/usr/bin/env python3
"""Convert a LeRobot dataset recorded by ``physical_ai_server`` into ICRT's format.

    python -m aiworker_icrt.convert_lerobot_icrt \
        --lerobot-root /data/ffw_sg2_mt \
        --out-dir /data/icrt/ffw_sg2_mt

What it does
------------
``physical_ai_server`` records 22-D joint-space vectors for both ``action`` (the
leader's command) and ``observation.state`` (the follower's achieved position),
plus three camera streams as mp4. ICRT wants a single hdf5 of per-episode groups
with Cartesian end-effector data, alongside three json sidecars.

The retargeting is the substance: every frame's 22-D joint vector is pushed
through forward kinematics into the 20-D dual-arm Cartesian representation
(``aiworker_icrt.constants``), so each arm looks like the single-arm DROID
end-effector space ICRT was pretrained on.

Why ``task`` matters
--------------------
``verb_to_episode.json`` groups episodes by task string, and that grouping is
what ICRT's ``sort_by_lang``/``task_barrier`` options use to build a training
sequence out of *several demonstrations of the same task*. Without it the prompt
masking degenerates and you are training plain behaviour cloning with extra
steps, not an in-context learner. Episodes whose task is missing or empty are
therefore reported rather than silently grouped together.

Storage
-------
Frames are stored decoded, as ICRT's loader expects random access by index.
At the default 180x320 that is ~518 KB per timestep across three cameras, so a
700-episode set of ~500-step episodes lands near 180 GB. Put it on NVMe: ICRT's
own TRAIN.md calls disk read speed the training bottleneck.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from aiworker_icrt import constants as C
from aiworker_icrt.kinematics import FFWKinematics

# hdf5 key names. 'image' must appear in the camera keys: ICRT's
# get_key_from_demo branches on that substring.
KEY_PROPRIO = 'observation/cartesian_position'
KEY_ACTION = 'action/cartesian_position'
KEY_IMAGE = {k: f'observation/image_{k}' for k in C.CAMERA_KEYS}


# --------------------------------------------------------------------------- #
# LeRobot reading
# --------------------------------------------------------------------------- #

def _read_jsonl(path: Path) -> List[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


class LeRobotReader:
    """Minimal reader for the LeRobot on-disk layout, v2.1 and v3.0.

    Deliberately does not import ``lerobot``: the library pulls a heavy dependency
    tree and its loader decodes video through torchcodec, while all we need is a
    few parquet columns and raw frames.

    The two layouts differ enough to matter:

    v2.1 (physical_ai_tools)  one parquet and one mp4 **per episode**, episode
                              metadata in meta/episodes.jsonl, camera features
                              named ``observation.images.<cam>``.

    v3.0 (cyclo_intelligence) many episodes **concatenated** into shared parquet
                              and mp4 files. Episode metadata lives in chunked
                              parquet under meta/episodes/ and carries the row
                              range (``dataset_from_index``/``dataset_to_index``)
                              and, for video, a **timestamp** range into the
                              shared mp4. Camera features are named
                              ``observation.images.rgb.<cam>``.
    """

    def __init__(self, root: Path, camera_map: Optional[Dict[str, str]] = None):
        self.root = root
        meta = root / 'meta'
        if not (meta / 'info.json').exists():
            raise FileNotFoundError(
                f'{root} does not look like a LeRobot dataset (no meta/info.json)'
            )
        self.info = json.loads((meta / 'info.json').read_text())
        self.version = str(self.info.get('codebase_version', 'v2.1'))
        self.fps = self.info.get('fps')
        self.chunks_size = self.info.get('chunks_size', 1000)
        self.is_v30 = self.version.startswith('v3')

        if self.is_v30:
            self.data_tmpl = self.info.get(
                'data_path', 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet')
            self.video_tmpl = self.info.get(
                'video_path',
                'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4')
            self.episodes = self._load_episodes_v30(meta)
            self.tasks = self._load_tasks_v30(meta)
        else:
            self.data_tmpl = self.info.get(
                'data_path',
                'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet')
            self.video_tmpl = self.info.get(
                'video_path',
                'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4')
            self.episodes = _read_jsonl(meta / 'episodes.jsonl')
            self.tasks = {t['task_index']: t['task']
                          for t in _read_jsonl(meta / 'tasks.jsonl')}

        self.camera_features = camera_map or self._discover_cameras()

    # -- schema discovery --------------------------------------------------- #

    def _discover_cameras(self) -> Dict[str, str]:
        """Map our canonical camera keys onto this dataset's feature names.

        Matching is by suffix so the same code handles ``observation.images.cam_head``
        (v2.1) and ``observation.images.rgb.cam_head`` (v3.0) without hardcoding
        either prefix.
        """
        feats = [k for k in self.info.get('features', {})
                 if k.startswith('observation.images')]
        mapping, unmatched = {}, list(feats)
        for cam in C.CAMERA_KEYS:
            # Accept either naming convention; see constants.CAMERA_ALIASES.
            names = C.CAMERA_ALIASES.get(cam, [cam])
            hit = [f for f in feats if f.split('.')[-1] in names]
            if hit:
                mapping[cam] = hit[0]
                if hit[0] in unmatched:
                    unmatched.remove(hit[0])
        if len(mapping) != len(C.CAMERA_KEYS):
            missing = [c for c in C.CAMERA_KEYS if c not in mapping]
            raise KeyError(
                f'could not find a camera feature for {missing} in {self.root}.\n'
                f'  dataset has: {feats}\n'
                f'  expected a feature whose last dotted segment is one of '
                f'{C.CAMERA_KEYS}.\n'
                f'  Pass --camera-map to override, e.g. '
                f'--camera-map cam_head=observation.images.rgb.head'
            )
        if unmatched:
            expected = [u for u in unmatched
                        if u.split('.')[-1] in C.CAMERA_IGNORED]
            other = [u for u in unmatched if u not in expected]
            if expected:
                print(f'  note: ignoring the ZED right eye {expected} — we use '
                      f'the left image as the single head view')
            if other:
                print(f'  note: ignoring unrecognised camera features {other}')
        return mapping

    # -- v3.0 metadata ------------------------------------------------------ #

    def _load_episodes_v30(self, meta: Path) -> List[dict]:
        import pyarrow.parquet as pq
        files = sorted((meta / 'episodes').rglob('*.parquet'))
        if not files:
            raise FileNotFoundError(f'{meta}/episodes/**/*.parquet not found')
        rows: List[dict] = []
        for f in files:
            rows.extend(pq.read_table(f).to_pylist())
        rows.sort(key=lambda r: r['episode_index'])
        return rows

    def _load_tasks_v30(self, meta: Path) -> Dict[int, str]:
        import pyarrow.parquet as pq
        path = meta / 'tasks.parquet'
        if not path.exists():
            return {}
        out = {}
        for r in pq.read_table(path).to_pylist():
            # Schema varies: task_index may be a column or the index.
            if 'task_index' in r and 'task' in r:
                out[r['task_index']] = r['task']
        return out

    # -- access ------------------------------------------------------------- #

    def episode_task(self, ep_meta: dict) -> Optional[str]:
        tasks = ep_meta.get('tasks') or []
        if isinstance(tasks, str):
            return tasks
        if len(tasks):
            return tasks[0]
        ti = ep_meta.get('task_index')
        return self.tasks.get(ti) if ti is not None else None

    def _v30_path(self, tmpl: str, chunk: int, file: int,
                  video_key: Optional[str] = None) -> Path:
        return self.root / tmpl.format(chunk_index=chunk, file_index=file,
                                       video_key=video_key)

    def read_frames(self, ep_meta: dict) -> Dict[str, np.ndarray]:
        """Return the parquet columns we need, already sliced to this episode."""
        import pyarrow.parquet as pq
        if self.is_v30:
            path = self._v30_path(self.data_tmpl,
                                  ep_meta['data/chunk_index'],
                                  ep_meta['data/file_index'])
            lo, hi = ep_meta['dataset_from_index'], ep_meta['dataset_to_index']
        else:
            ep = ep_meta['episode_index']
            path = self.root / self.data_tmpl.format(
                episode_chunk=ep // self.chunks_size, episode_index=ep)
            lo = hi = None

        if not path.exists():
            raise FileNotFoundError(f'missing data file {path}')
        tbl = pq.read_table(path)
        if lo is not None:
            # v3.0 packs many episodes per file; take only this episode's rows.
            tbl = tbl.slice(lo, hi - lo)

        cols = set(tbl.column_names)
        out = {}
        for name in ('action', 'observation.state'):
            if name not in cols:
                raise KeyError(
                    f'episode {ep_meta["episode_index"]}: parquet has no {name!r}; '
                    f'found {sorted(cols)}')
            out[name] = np.stack(tbl.column(name).to_pylist()).astype(np.float64)
        return out

    def read_video(self, ep_meta: dict, cam_key: str, size) -> np.ndarray:
        """Decode one camera's frames for this episode as (T, H, W, 3) uint8 RGB."""
        import cv2
        feature = self.camera_features[cam_key]
        n_expected = int(ep_meta.get('length') or 0)

        if self.is_v30:
            path = self._v30_path(
                self.video_tmpl,
                ep_meta[f'videos/{feature}/chunk_index'],
                ep_meta[f'videos/{feature}/file_index'],
                video_key=feature)
            from_ts = float(ep_meta[f'videos/{feature}/from_timestamp'])
        else:
            ep = ep_meta['episode_index']
            path = self.root / self.video_tmpl.format(
                episode_chunk=ep // self.chunks_size, episode_index=ep,
                video_key=feature)
            from_ts = 0.0

        if not path.exists():
            raise FileNotFoundError(f'missing video {path}')

        cap = cv2.VideoCapture(str(path))
        H, W = size
        frames = []
        try:
            if from_ts > 0.0:
                # Seek by time into the concatenated file. Keyframe seeking is
                # approximate, so we then take exactly `length` frames rather
                # than reading until to_timestamp -- an off-by-a-frame drift
                # would silently misalign video against the parquet rows.
                cap.set(cv2.CAP_PROP_POS_MSEC, from_ts * 1000.0)
            while n_expected == 0 or len(frames) < n_expected:
                ok, frame = cap.read()
                if not ok:
                    break
                if (frame.shape[0], frame.shape[1]) != (H, W):
                    frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
                frames.append(frame[:, :, ::-1])       # BGR -> RGB
        finally:
            cap.release()

        if not frames:
            raise ValueError(f'decoded zero frames from {path}')
        if n_expected and len(frames) < n_expected:
            raise ValueError(
                f'{path}: got {len(frames)} frames, episode declares {n_expected}. '
                f'Video and parquet would be misaligned.')
        return np.ascontiguousarray(np.stack(frames), dtype=np.uint8)


# --------------------------------------------------------------------------- #
# Conversion
# --------------------------------------------------------------------------- #

def convert(
    lerobot_root: Path,
    out_dir: Path,
    image_size=(180, 320),
    urdf: Optional[str] = None,
    hdf5_name: str = 'ffw_sg2.hdf5',
    limit: Optional[int] = None,
    prompt_dir: Optional[Path] = None,
    camera_map: Optional[Dict[str, str]] = None,
    target_fps: Optional[float] = None,
    episode_prefix: str = '',
    append: bool = False,
    action_source: str = 'leader',
) -> None:
    import h5py
    try:
        from tqdm import tqdm
    except ImportError:                     # optional
        def tqdm(x, **k):
            return x

    reader = LeRobotReader(lerobot_root, camera_map=camera_map)
    kin = FFWKinematics(urdf) if urdf else FFWKinematics()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ICRT was built around ~15 fps recordings: its default episode-length filter
    # is 450 frames, its KV cache is budgeted in (state, action) pairs, and its
    # 16-step action chunk spans ~1s at that rate. A 30 fps recording halves all
    # of those in wall-clock terms, so subsample rather than silently training on
    # a different temporal scale.
    stride = 1
    if target_fps and reader.fps:
        stride = max(1, int(round(reader.fps / target_fps)))
        if stride > 1:
            print(f'  subsampling {reader.fps} -> {reader.fps / stride:g} fps '
                  f'(every {stride} frames)')

    episodes = reader.episodes[:limit] if limit else reader.episodes
    print(f'{len(episodes)} episodes in {lerobot_root} '
          f'(codebase_version={reader.version}, fps={reader.fps})')
    for cam, feat in reader.camera_features.items():
        print(f'  {cam:16} <- {feat}')

    epi_len: Dict[str, int] = {}
    verb_to_episode: Dict[str, List[str]] = defaultdict(list)
    hdf5_keys: List[str] = []
    # (replaced below when --append finds an existing set)
    untasked: List[str] = []
    skipped: List[str] = []

    h5_path = out_dir / hdf5_name

    # Appending lets several raw datasets accumulate into ONE converted set,
    # which is what the multi-task story needs: verb_to_episode has to group
    # episodes of different tasks in a single file for ICRT's prompt masking to
    # mean anything, and for an evaluation to prompt from one task while
    # replaying another.
    if append and h5_path.exists():
        epi_len = json.loads((out_dir / 'epi_len_mapping.json').read_text())
        hdf5_keys = json.loads((out_dir / 'hdf5_keys.json').read_text())
        verb_to_episode = defaultdict(
            list, json.loads((out_dir / 'verb_to_episode.json').read_text()))
        print(f'  appending to {len(hdf5_keys)} existing episodes')

    with h5py.File(h5_path, 'a' if append else 'w') as h5:
        for ep_meta in tqdm(episodes, desc='converting'):
            ep = ep_meta['episode_index']
            # 'episode' in the name keeps us compatible with ICRT's fallback
            # image-shape heuristic even if the array path is ever bypassed.
            # Every LeRobot set numbers from episode_000000, so a prefix is
            # mandatory when merging or the second dataset silently overwrites
            # the first.
            name = f'{episode_prefix}episode_{ep:06d}'
            if name in h5:
                print(f'  skipping {name}: already present', file=sys.stderr)
                skipped.append(name)
                continue
            try:
                cols = reader.read_frames(ep_meta)
                images = {k: reader.read_video(ep_meta, k, image_size)
                          for k in C.CAMERA_KEYS}
            except (FileNotFoundError, KeyError, ValueError) as exc:
                print(f'  skipping {name}: {exc}', file=sys.stderr)
                skipped.append(name)
                continue

            action_j = cols['action'][::stride]
            state_j = cols['observation.state'][::stride]

            # ACTION SOURCE. `action` is the LEADER's command; `observation.state`
            # is what the follower achieved. They differ by the servo lag -- and
            # that lag is a property of the controller tuning on the day, not of
            # the task.
            #
            # Measured across our three recording sessions:
            #     box    |action - state|  0.006736 rad   21.1 mm at the hand
            #     ecu                      0.009399 rad   38.4 mm
            #     ylw2                     0.000223 rad    0.75 mm   <- 30-40x less
            #
            # ICRT's target is action[t+j] - proprio[t]. With the leader source
            # that target is ~21-38 mm of servo lag for two tasks and ~0.75 mm for
            # the third, so "action" means something different depending on which
            # session recorded it. The model cannot tell the regimes apart and
            # emits a roughly constant ~22 mm delta everywhere -- correct for box,
            # 55x too large for ylw2.
            #
            # 'achieved' takes the target from where the arm actually went, one
            # step ahead, so the label is real motion and is independent of how
            # well the controller was tracking. It is also the thing a policy
            # should output: the next pose, leaving the servo to close the gap.
            if action_source == 'achieved':
                action_j = np.concatenate([state_j[1:], state_j[-1:]], axis=0)
            if stride > 1:
                images = {k: v[::stride] for k, v in images.items()}
            T = min(len(action_j), len(state_j), *(len(v) for v in images.values()))
            if T < 2:
                print(f'  skipping {name}: only {T} usable frames', file=sys.stderr)
                skipped.append(name)
                continue
            if len({len(action_j), len(state_j), *(len(v) for v in images.values())}) > 1:
                # Video and parquet lengths can differ by a frame at episode
                # boundaries; truncating to the shortest keeps them aligned.
                print(f'  {name}: trimming streams to {T} frames')

            for arr_name, arr in (('action', action_j), ('state', state_j)):
                if arr.shape[1] < C.ARMS_ONLY_DIM:
                    raise ValueError(
                        f'{name}: {arr_name} is {arr.shape[1]}-D, below the '
                        f'{C.ARMS_ONLY_DIM} needed for both arms + grippers. '
                        f'Recording layouts of {C.ARMS_ONLY_DIM} (arms only, e.g. '
                        f'ffw_bg2_rev4_custom / ffw_arm_only) and {C.JOINT_DIM} '
                        f'(full ffw_sg2) are supported.'
                    )

            grp = h5.create_group(name)
            # proprio 23 = arms(20) + head(2) + lift(1)   -- observed
            # action  21 = arms(20) + lift(1)             -- commanded
            grp.create_dataset(
                KEY_PROPRIO, data=kin.joints_to_proprio(state_j[:T]).astype(np.float32))
            grp.create_dataset(
                KEY_ACTION, data=kin.joints_to_action(action_j[:T]).astype(np.float32))
            # Keep the raw joints too, for BOTH state and action.
            #
            # The 20-D Cartesian vectors cover only the arms, so without these
            # the head/lift/base information is lost at conversion. It is needed
            # to seed IK, to reproduce the uncontrolled pose when replaying a
            # prompt, and — the reason the action copy matters — to recover the
            # COMMANDED setpoint for those joints rather than the measured state.
            #
            # On the ECU recordings the head was commanded to a single constant
            # (0.6427, 0.3496) while the measured value jittered +/-0.18 deg
            # around it. The setpoint is the honest preflight target: the head
            # carries cam_head, so pointing it somewhere else at inference puts
            # the policy's main camera off-distribution with no error raised.
            grp.create_dataset('observation/joint_position',
                               data=state_j[:T].astype(np.float32))
            grp.create_dataset('action/joint_position',
                               data=action_j[:T].astype(np.float32))
            for cam in C.CAMERA_KEYS:
                grp.create_dataset(
                    KEY_IMAGE[cam], data=images[cam][:T],
                    chunks=(1, *image_size, 3), compression='lzf')

            epi_len[name] = T
            hdf5_keys.append(name)
            task = reader.episode_task(ep_meta)
            if task:
                verb_to_episode[task].append(name)
            else:
                untasked.append(name)

            if prompt_dir is not None:
                prompt_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    prompt_dir / f'{name}.npz',
                    images=np.stack([images[c][:T] for c in C.CAMERA_KEYS], axis=1),
                    joint_states=state_j[:T].astype(np.float32),
                    joint_actions=action_j[:T].astype(np.float32),
                    actions_cartesian=kin.joints_to_action(action_j[:T]).astype(np.float32),
                    task=task or '',
                )

    # --- sidecars ---
    (out_dir / 'epi_len_mapping.json').write_text(json.dumps(epi_len, indent=2))
    (out_dir / 'hdf5_keys.json').write_text(json.dumps(hdf5_keys, indent=2))
    (out_dir / 'verb_to_episode.json').write_text(
        json.dumps(dict(verb_to_episode), indent=2))

    dataset_config = {
        'dataset_path': [str(h5_path)],
        'hdf5_keys': [str(out_dir / 'hdf5_keys.json')],
        'epi_len_mapping_json': str(out_dir / 'epi_len_mapping.json'),
        'verb_to_episode': str(out_dir / 'verb_to_episode.json'),
        'train_split': 0.95,
        'action_keys': [KEY_ACTION],
        'image_keys': [KEY_IMAGE[c] for c in C.CAMERA_KEYS],
        'proprio_keys': [KEY_PROPRIO],
        # Proprio/action are already [xyz, rot_6d, gripper] per arm; tell the
        # loader not to reinterpret dims 3: as a single euler/quat rotation.
        'skip_rot_conversion': True,
        # Recorded for downstream readers; the arm count cannot be inferred from
        # the vector width once non-arm channels are appended.
        'num_arms': 2,
        'proprio_extra': C.PROPRIO_EXTRA,
        'action_extra': C.ACTION_EXTRA,
    }
    (out_dir / 'dataset_config.json').write_text(json.dumps(dataset_config, indent=2))

    # --- report ---
    total = sum(epi_len.values())
    if epi_len:
        lo, hi = min(epi_len.values()), max(epi_len.values())
        print(f'  episode length {lo}..{hi} frames')
        if hi > 450:
            print(f'  NOTE: ICRT\'s default episode-length filter is '
                  f'[30, 450] frames and would drop episodes longer than 450. '
                  f'Pass --dataset-cfg.maximum-length {hi + 50} when training, '
                  f'or re-run with --target-fps to subsample.')
        if len(hdf5_keys) < 4:
            print(f'  NOTE: only {len(hdf5_keys)} episodes. ICRT requires at '
                  f'least min_demos=4 per task to survive the train/val split, '
                  f'so this will yield an empty dataset.')
    print(f'\nwrote {len(hdf5_keys)} episodes / {total} frames -> {h5_path}')
    print(f'  {len(verb_to_episode)} tasks: ' + ', '.join(
        f'{k}({len(v)})' for k, v in sorted(
            verb_to_episode.items(), key=lambda kv: -len(kv[1]))[:10]))
    if skipped:
        print(f'  SKIPPED {len(skipped)}: {skipped[:5]}{"..." if len(skipped) > 5 else ""}')
    if untasked:
        print(f'  WARNING: {len(untasked)} episodes have no task string and are '
              f'absent from verb_to_episode.json. ICRT groups demonstrations by '
              f'task to build in-context prompts; these will never be sampled as '
              f'same-task context. {untasked[:5]}')
    if len(verb_to_episode) < 2:
        print('  WARNING: fewer than 2 distinct tasks. In-context learning has '
              'nothing to generalise across -- check the task strings.')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--lerobot-root', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--image-size', type=int, nargs=2, default=(180, 320),
                    metavar=('H', 'W'),
                    help='stored frame size (default matches ICRT-MT)')
    ap.add_argument('--urdf', default=None, help='override the bundled FFW-SG2 URDF')
    ap.add_argument('--hdf5-name', default='ffw_sg2.hdf5')
    ap.add_argument('--limit', type=int, default=None,
                    help='convert only the first N episodes (for a smoke test)')
    ap.add_argument('--prompt-dir', type=Path, default=None,
                    help='also write per-episode .npz demos for the ROS node to '
                         'prompt from')
    ap.add_argument('--episode-prefix', default='',
                    help='prepended to episode names; required when merging '
                         'datasets, which all number from episode_000000')
    ap.add_argument('--append', action='store_true',
                    help='add to an existing converted set instead of replacing it')
    ap.add_argument('--action-source', choices=('leader', 'achieved'),
                    default='leader',
                    help="'leader' uses the recorded command (original behaviour); "
                         "'achieved' uses the follower pose one step ahead, which "
                         "removes per-session servo lag from the target")
    ap.add_argument('--target-fps', type=float, default=None,
                    help='subsample to approximately this rate (ICRT assumes ~15)')
    ap.add_argument('--camera-map', nargs='*', default=None,
                    metavar='CANONICAL=FEATURE',
                    help='override camera feature discovery, e.g. '
                         'cam_head=observation.images.rgb.head')
    a = ap.parse_args()
    cam_map = None
    if a.camera_map:
        cam_map = dict(kv.split('=', 1) for kv in a.camera_map)
    convert(a.lerobot_root, a.out_dir, tuple(a.image_size), a.urdf,
            a.hdf5_name, a.limit, a.prompt_dir, cam_map, a.target_fps,
            a.episode_prefix, a.append, a.action_source)


if __name__ == '__main__':
    main()
