# Docker notes

Owned by `aiworker-iclr-61`. Built via `../container.sh`, which picks the
Dockerfile from `uname -m` and passes `BASE_IMAGE` / `ROS_DISTRO` /
`INSTALL_DATA_DEPS` through `docker-compose.yml`.

## A green build does NOT mean the GPU works

The build asserts that torch is CUDA-capable, but `docker build` has no GPU, so
it can only check the binary — never that a GPU initialises. **`../container.sh
status` is the real gate**: it runs `torch.cuda.is_available()` and prints the
device capability inside the running container. Do not report the stack as
working until that passes.

What the build-time guard *can* catch, and does:

| check | catches |
|---|---|
| `torch.version.cuda` non-empty | a PyPI CPU wheel replaced NVIDIA's build |
| `torch.cuda.get_arch_list()` non-empty | a **CUDA-flavoured CPU build** — see below |
| arch list contains `sm_110` + `sm_120` | a CUDA torch with no Blackwell kernels |

The middle row exists because of a live counterexample on this robot:
`robotis/cyclo-intelligence:1.4.0` reports `torch 2.7.0 / cuda 12.8` — it would
sail past a version-only assert — but `cuda.is_available()` is `False` and the
first allocation dies with "Found no NVIDIA driver".

Measured inside that image **with `--gpus all`**:

```
version.cuda         : 12.8
backends.is_built    : True       <-- would NOT catch it
_cuda_getArchFlags() : 'sm_87'    <-- the actual cause
get_arch_list()      : []         <-- a SYMPTOM, not the cause
is_available()       : False
```

**It is compiled for sm_87 only — Jetson Orin.** It cannot drive GB10 (sm_121),
so `is_available()` is False, and `get_arch_list()` returns `[]` *because* of
that. The empty list is downstream of the real problem. This is precisely the
L4T/JetPack-vs-SBSA mismatch the Dockerfile headers warn about, in ROBOTIS' own
image.

**Read the raw flags, not `get_arch_list()`.** `torch.cuda.get_arch_list()` is
gated on `is_available()` in NGC's torch 2.10.0a0 and returns `[]` whenever no
GPU is visible — which is every `docker build` layer. Measured on the untouched
base image:

```
get_arch_list()      : []
_cuda_getArchFlags() : 'sm_80 sm_86 sm_90 sm_100 sm_110 sm_120 compute_120'
```

A guard built on `get_arch_list()` would fail 100% of builds on a perfectly good
image. `torch._C._cuda_getArchFlags()` reads the string out of the binary and
works with no GPU present, which is what makes it usable at build time. Note
`compute_120` is present — the `+PTX` in `TORCH_CUDA_ARCH_LIST` does surface,
and that is the PTX GB10's sm_121 JITs from.

## PIP_CONSTRAINT: blast radius is the image only

The build generates `/etc/pip-constraints.txt` from the as-shipped versions of
torch / torchvision / numpy / scipy / opencv / pillow and sets `PIP_CONSTRAINT`
to it, so any later `pip install` **inside the image** — including one you run
by hand — errors rather than silently swapping NVIDIA's CUDA build for a PyPI
wheel.

It governs nothing else. A scratch venv, a conda env, a `pip --user` on the
host, or a sibling container are all invisible to it. A green build says nothing
about the soundness of your host Python environment.

If a package fights the constraint, **do not relax it** — pin the offending
package properly, or move it into `requirements-data.txt` behind
`INSTALL_DATA_DEPS=1`.

## GPU access on this machine: CDI, not the nvidia runtime

`docker info` on this box registers only `runc` and `io.containerd.runc.v2` —
**there is no `nvidia` runtime**. GPU passthrough goes through CDI
(nvidia-container-toolkit 1.20.0, spec at `/var/run/cdi/nvidia.yaml`, spec dirs
`/etc/cdi` and `/var/run/cdi`).

Docker 29 resolves `--gpus all` through CDI, and compose's
`deploy.resources.reservations.devices.driver: nvidia` — what
`docker-compose.yml` uses — resolves through it too. **Verified**: a compose
service with exactly that reservation block reported `NVIDIA GB10, 12.1`.

Anything telling you to pass `--runtime nvidia` is wrong for this machine.
Fallback, should the reservation block ever stop resolving:
`devices: ["nvidia.com/gpu=all"]`.

## Docker permissions

`docker info` may fail while `getent group docker` shows your user — the login
session's process credentials predate the group grant. `container.sh` handles
this by re-execing itself under `sg docker`. A fresh login (or `newgrp docker`)
fixes it permanently. This is *not* a missing group membership, and `usermod` is
not the fix.

## Base image

`nvcr.io/nvidia/pytorch:25.11-py3` — CUDA 13.0.2, torch 2.10.0a0, Ubuntu 24.04
noble, Python 3.12, multi-arch (amd64 + arm64). Chosen because it is the newest
tag built against a driver at or below this host's 580.173.02, and because its
`TORCH_CUDA_ARCH_LIST` covers sm_110 (Thor) natively and sm_120+PTX (GB10
sm_121 JITs from it). Floor is `25.08-py3`; anything older lacks sm_110/sm_120.

---

## Check behaviour, not presence

Every green check in this stack that later turned out to be hiding a real
failure was a check that confirmed something was **present** rather than that it
**worked**. Collected here because the pattern repeated five times in one night,
and each time the passing check was the reason nobody looked further.

| the check that passed | what it actually hid |
|---|---|
| `import cv2` | apt cv2 4.6 built against numpy 1.x — present on `sys.path`, cannot load |
| `import cv_bridge` | conversion path segfaults on first use (numpy ABI) or raises `KeyError: 16` (OpenCV 5 type shift) |
| `torch.version.cuda` non-empty | a torch compiled for `sm_87` only — genuinely CUDA-built, useless on this GPU |
| `backends.cuda.is_built()` | same; returns `True` for that torch |
| `torch.cuda.get_arch_list()` **with** a GPU | returns `[]` without one on some builds — a guard built on it fails 100% of builds |
| `ros2 node list` failing → "no nodes / no router" | ROS not on `PATH` at all; the healthy and broken states printed identically |

The replacements all exercise the thing end to end:

- cv2 → `imdecode` a real buffer, `resize` a real array, construct a `VideoCapture`
- cv_bridge → round-trip an array through `cv2_to_imgmsg` and back (this is what
  found both failures; `import` found neither)
- torch → `_cuda_getArchFlags()` and assert a Blackwell target is in it
- ROS → three distinct states, plus an independent `import rclpy`, so "env
  missing" can never print as "waiting for a router"

When adding a check here, ask what a broken version of the thing would print. If
the answer is "the same thing", the check is worse than nothing: it converts an
absence of information into confidence.
