#!/usr/bin/env python3
"""Visualise a converted ICRT dataset -- the retargeting is the part worth seeing.

    python -m aiworker_icrt.visualize --dataset data/icrt_ecu --episode episode_000000

Writes, per episode:
  <out>/<episode>.mp4   the three camera streams tiled, with a live readout of
                        both end-effector poses and gripper states
  <out>/<episode>.png   end-effector xyz traces, gripper signals, and a 3-D path

The point is to catch retargeting errors that pass every numeric check: a frame
convention off by a sign, a left/right swap, a gripper inverted. Those look fine
in an assertion and obvious in a picture.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np

from aiworker_icrt import constants as C


def _episode_names(h5: h5py.File, want: Optional[List[str]], limit: int) -> List[str]:
    names = sorted(h5.keys())
    if want:
        missing = [w for w in want if w not in names]
        if missing:
            raise KeyError(f'no such episode(s): {missing}. Have {names[:5]}...')
        return want
    return names[:limit]


class _Mp4Writer:
    """Write browser-playable H.264, falling back to OpenCV only if forced.

    cv2.VideoWriter cannot produce H.264 here: opencv-python-headless ships an
    FFmpeg without libx264 (GPL), so every avc1/H264/X264 fourcc fails to OPEN --
    measured, not assumed. Only 'mp4v' succeeds, and it yields MPEG-4 Part 2,
    which NO browser will play. The file is perfectly valid and plays in VLC and
    ffprobe, which is exactly why it slipped through: it looked correct
    everywhere except the one place it is consumed.

    So we pipe raw frames to the system ffmpeg, which does have libx264.
    `-movflags +faststart` moves the moov atom to the head of the file so the
    browser can begin playing before the download finishes -- without it a
    900-frame episode buffers completely first, which reads as "still broken".
    """

    def __init__(self, path: Path, size, fps: float):
        w, h = size
        # yuv420p needs even dimensions; odd sizes make ffmpeg fail outright.
        self.pad_w, self.pad_h = w % 2, h % 2
        self.w, self.h = w + self.pad_w, h + self.pad_h
        self.path = path
        self.proc = None
        self.cv_writer = None

        if shutil.which('ffmpeg'):
            self.proc = subprocess.Popen(
                ['ffmpeg', '-y', '-loglevel', 'error',
                 '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                 '-s', f'{self.w}x{self.h}', '-r', str(fps), '-i', '-',
                 '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                 '-preset', 'fast', '-crf', '23',
                 '-movflags', '+faststart', str(path)],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE)
        else:
            import cv2
            print('  WARNING: ffmpeg not found — falling back to mp4v, which '
                  'browsers cannot play. Install ffmpeg for a usable video.')
            self.cv_writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (self.w, self.h))

    def write(self, frame) -> None:
        if self.pad_w or self.pad_h:
            frame = np.pad(frame, ((0, self.pad_h), (0, self.pad_w), (0, 0)))
        if self.proc is not None:
            self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        else:
            self.cv_writer.write(frame)

    def close(self) -> None:
        if self.proc is not None:
            self.proc.stdin.close()
            if self.proc.wait() != 0:
                raise RuntimeError(
                    f'ffmpeg failed writing {self.path}: '
                    f'{self.proc.stderr.read().decode()[:400]}')
        else:
            self.cv_writer.release()


def render_video(h5: h5py.File, ep: str, out: Path, fps: float) -> Path:
    import cv2
    g = h5[ep]
    cams = [g[f'observation/image_{k}'][:] for k in C.CAMERA_KEYS]
    cart = g['observation/cartesian_position'][:]
    T = min(len(cart), *(len(c) for c in cams))

    # Tile the cameras on one row, normalising heights.
    h = max(c.shape[1] for c in cams)

    def _fit(img):
        s = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * s), h))

    widths = [_fit(c[0]).shape[1] for c in cams]
    strip_w = sum(widths)
    panel = 96                                   # readout strip under the video

    path = out / f'{ep}.mp4'
    vw = _Mp4Writer(path, (strip_w, h + panel), fps)
    try:
        for t in range(T):
            row = np.hstack([_fit(c[t]) for c in cams])[:, :, ::-1]   # RGB->BGR
            frame = np.zeros((h + panel, strip_w, 3), np.uint8)
            frame[:h] = row
            frame[h:] = (18, 16, 14)          # near-black strip, faintly warm

            # Camera labels: small, bottom-left of each tile, on a translucent
            # plate so they stay legible over both bright and dark scenes. The
            # charts below carry the numbers, so the video only needs to say
            # WHICH view you are looking at.
            for i, key in enumerate(C.CAMERA_KEYS):
                x = sum(widths[:i])
                (tw, th), _ = cv2.getTextSize(key, cv2.FONT_HERSHEY_DUPLEX, 0.42, 1)
                y0 = h - 14 - th
                plate = frame[y0 - 5:y0 + th + 7, x + 10:x + 10 + tw + 14]
                plate[:] = (plate * 0.35).astype(np.uint8)
                cv2.putText(frame, key, (x + 17, h - 16),
                            cv2.FONT_HERSHEY_DUPLEX, 0.42, (232, 230, 226), 1,
                            cv2.LINE_AA)
                if i:                          # hairline between tiles
                    frame[:h, x:x + 1] = (40, 38, 36)

            # Readout strip. Restrained: one line per arm, dim labels, bright
            # values, and a thin gripper bar. Deliberately quieter than the
            # charts underneath it, which are the real instrument.
            base_y = h + 26
            for i, (side, sl, accent) in enumerate(
                    # BGR, not RGB: teal for the left arm, amber for the right.
                    (('L', C.SLICE_CART_L, (150, 190, 90)),
                     ('R', C.SLICE_CART_R, (80, 160, 225)))):
                b = cart[t, sl]
                p_, grip = b[C.SLICE_BLOCK_POS], float(b[C.BLOCK_GRIPPER])
                y = base_y + i * 26
                cv2.putText(frame, side, (18, y), cv2.FONT_HERSHEY_DUPLEX,
                            0.46, accent, 1, cv2.LINE_AA)
                for k, (lbl, val) in enumerate(zip('xyz', p_)):
                    xx = 46 + k * 104
                    cv2.putText(frame, lbl, (xx, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.38, (120, 118, 114), 1, cv2.LINE_AA)
                    cv2.putText(frame, f'{val:+.3f}', (xx + 12, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                                (224, 222, 218), 1, cv2.LINE_AA)
                # gripper: hairline track, accent fill, value at the end
                gx, gw = 380, 96
                cv2.rectangle(frame, (gx, y - 9), (gx + gw, y - 1), (52, 50, 48), -1)
                cv2.rectangle(frame, (gx, y - 9),
                              (gx + int(gw * grip), y - 1), accent, -1)
                cv2.putText(frame, f'{grip:.2f}', (gx + gw + 10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 148, 144), 1,
                            cv2.LINE_AA)

            # Progress hairline along the very bottom, plus a quiet frame count.
            cv2.rectangle(frame, (0, h + panel - 2),
                          (int(strip_w * (t + 1) / T), h + panel), (150, 190, 90), -1)
            cv2.putText(frame, f'{t + 1}/{T}', (strip_w - 74, h + 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 118, 114), 1,
                        cv2.LINE_AA)
            vw.write(frame)
    finally:
        vw.close()
    return path


def render_plot(h5: h5py.File, ep: str, out: Path, fps: float) -> Path:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    g = h5[ep]
    cart = g['observation/cartesian_position'][:]
    act = g['action/cartesian_position'][:]
    t = np.arange(len(cart)) / fps

    fig = plt.figure(figsize=(15, 8))
    fig.suptitle(f'{ep} - end-effector in {C.FK_ROOT_LINK}', fontsize=13)

    for col, (side, sl) in enumerate((('left', C.SLICE_CART_L),
                                      ('right', C.SLICE_CART_R))):
        p = cart[:, sl][:, C.SLICE_BLOCK_POS]
        pa = act[:, sl][:, C.SLICE_BLOCK_POS]

        ax = fig.add_subplot(2, 3, col * 3 + 1)
        for i, lbl in enumerate('xyz'):
            ax.plot(t, p[:, i], label=lbl, lw=1.3)
            ax.plot(t, pa[:, i], lw=0.8, ls='--', alpha=0.5)
        ax.set_title(f'{side} eef position (solid=state, dashed=action)', fontsize=9)
        ax.set_xlabel('s')
        ax.set_ylabel('m')
        ax.legend(fontsize=7)
        ax.grid(alpha=.3)

        ax = fig.add_subplot(2, 3, col * 3 + 2)
        ax.plot(t, cart[:, sl][:, C.BLOCK_GRIPPER], color='tab:green', lw=1.3)
        ax.set_title(f'{side} gripper (0=open, 1=closed)', fontsize=9)
        ax.set_xlabel('s')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=.3)

        ax = fig.add_subplot(2, 3, col * 3 + 3, projection='3d')
        ax.plot(p[:, 0], p[:, 1], p[:, 2], lw=1.2)
        ax.scatter(*p[0], c='g', s=30, label='start')
        ax.scatter(*p[-1], c='r', s=30, label='end')
        dist = np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
        ax.set_title(f'{side} path ({dist:.2f} m)', fontsize=9)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.legend(fontsize=7)

    fig.tight_layout()
    path = out / f'{ep}.png'
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def export_series(h5: h5py.File, ep: str, out: Path, fps: float) -> Path:
    """Dump the per-frame series as JSON so a UI can chart it against the video.

    Modelled on cyclo_intelligence's replay view: the chart and the video share
    one playhead, so the series must carry a `time` in seconds that lines up with
    the mp4's own timebase. State and action are emitted as separate series to be
    overlaid, which is what makes a tracking error visible at a glance.
    """
    g = h5[ep]
    state = g['observation/cartesian_position'][:]
    action = g['action/cartesian_position'][:]
    joints = g['observation/joint_position'][:]
    T = len(state)

    def _arm(a, sl):
        b = a[:, sl]
        return {
            'x': b[:, 0].round(5).tolist(),
            'y': b[:, 1].round(5).tolist(),
            'z': b[:, 2].round(5).tolist(),
            'rot6d': b[:, C.SLICE_BLOCK_ROT6D].round(5).tolist(),
            'gripper': b[:, C.BLOCK_GRIPPER].round(5).tolist(),
        }

    doc = {
        'episode': ep,
        'frames': int(T),
        'fps': fps,
        'frame_root': C.FK_ROOT_LINK,
        'gripper_sense': '0 = open, 1 = closed (RH-P12-RN)',
        'time': [round(t / fps, 4) for t in range(T)],
        'cameras': list(C.CAMERA_KEYS),
        'state': {'left': _arm(state, C.SLICE_CART_L), 'right': _arm(state, C.SLICE_CART_R)},
        'action': {'left': _arm(action, C.SLICE_CART_L), 'right': _arm(action, C.SLICE_CART_R)},
        'joints': {
            'names': C.JOINT_ORDER[:joints.shape[1]],
            'values': joints.round(5).tolist(),
        },
    }
    path = out / f'{ep}.json'
    path.write_text(json.dumps(doc))
    return path


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', type=Path, required=True,
                    help='converted output dir (contains ffw_sg2.hdf5)')
    ap.add_argument('--hdf5-name', default='ffw_sg2.hdf5')
    ap.add_argument('--out', type=Path, default=None, help='default: <dataset>/viz')
    ap.add_argument('--episode', nargs='*', default=None,
                    help='episode names, e.g. episode_000000')
    ap.add_argument('--limit', type=int, default=1)
    ap.add_argument('--fps', type=float, default=15.0)
    ap.add_argument('--no-video', action='store_true')
    ap.add_argument('--no-plot', action='store_true')
    ap.add_argument('--no-json', action='store_true',
                    help='skip the per-frame JSON series used by the UI')
    a = ap.parse_args()

    out = a.out or (a.dataset / 'viz')
    out.mkdir(parents=True, exist_ok=True)
    with h5py.File(a.dataset / a.hdf5_name, 'r') as h5:
        for ep in _episode_names(h5, a.episode, a.limit):
            n = len(h5[ep]['observation/cartesian_position'])
            print(f'{ep}: {n} frames')
            if not a.no_plot:
                print(f'  plot  -> {render_plot(h5, ep, out, a.fps)}')
            if not a.no_video:
                print(f'  video -> {render_video(h5, ep, out, a.fps)}')
            if not a.no_json:
                print(f'  json  -> {export_series(h5, ep, out, a.fps)}')


if __name__ == '__main__':
    main()
