"""Job-launcher web server for the ICRT x AI Worker pipeline.

Four actions — convert, visualize, train, evaluate — each of which shells out to
a CLI that already exists and streams its output back to the browser.

Python standard library only, deliberately. This is a job launcher and a log
tail; adding a web framework to an image built around torch (and guarded by
PIP_CONSTRAINT) is a bad trade. No build step on the front end either.

Subprocesses rather than imports: a traceback in the converter kills a job, not
the server, and the failure is visible in exactly the form it would take on the
command line.

    python -m aiworker_icrt.ui [--port 8770] [--host 127.0.0.1]

Binds loopback by default. This box is on the robot network and the UI can start
GPU jobs, so --host must be passed explicitly to expose it.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

REPO = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).resolve().parent / "static"
ICRT = REPO / "third_party" / "icrt"

# Minimum episodes per task before the train/val split swallows it whole.
MIN_DEMOS = 4
# seq-length must EXCEED the longest episode, or --no-prompt-loss leaves most
# windows without an episode boundary and the loss reads a flat 0.0000 while
# training "succeeds". The threshold is a property of the DATA, not a constant:
# 512 happens to clear the 280-412 frame ECU sample, but the user's real
# episodes are 60 s -> 1800 frames at 30 fps -> 900 subsampled to 15 fps, where
# 512 would silently fail in exactly the same way. Only used as a floor for the
# prefill when a dataset has no length data.
MIN_SEQ_LENGTH = 512
LOSS_RE = re.compile(r"loss:\s*([0-9.]+)")


# --------------------------------------------------------------------------- #
#  jobs
# --------------------------------------------------------------------------- #
class Job:
    """One subprocess, its merged output, and whatever we can parse from it."""

    def __init__(self, kind: str, cmd: list[str], cwd: Path, env: dict):
        self.id = f"{kind}-{uuid.uuid4().hex[:8]}"
        self.kind = kind
        self.cmd = cmd
        self.lines: list[str] = []
        self.losses: list[float] = []
        self.status = "running"
        self.returncode: int | None = None
        self.started = time.time()
        self.ended: float | None = None
        self._lock = threading.Lock()

        self._append(f"$ {' '.join(shlex.quote(c) for c in cmd)}")
        self._append(f"  cwd={cwd}")
        self.proc = subprocess.Popen(
            cmd, cwd=str(cwd), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, start_new_session=True,
        )
        threading.Thread(target=self._pump, daemon=True).start()

    def _append(self, line: str) -> None:
        with self._lock:
            self.lines.append(line)
            # Keep memory bounded on a long training run without losing the head,
            # which is where the config and the warnings are.
            if len(self.lines) > 20000:
                del self.lines[200:1200]

    def _pump(self) -> None:
        assert self.proc.stdout is not None
        for raw in self.proc.stdout:
            line = raw.rstrip("\n")
            self._append(line)
            m = LOSS_RE.search(line)
            if m:
                try:
                    self.losses.append(float(m.group(1)))
                except ValueError:
                    pass
        self.proc.wait()
        self.returncode = self.proc.returncode
        self.status = "done" if self.returncode == 0 else "failed"
        # A training run that exits 0 having produced no loss samples did not
        # train. Exit status only says the process ended tidily; it says nothing
        # about whether any work happened. This catches 0-epoch runs and any
        # future variant of "finished successfully, did nothing".
        if self.kind == "train" and self.status == "done" and not self.losses:
            self.status = "failed"
            self._append("[no loss samples recorded — this run trained nothing. "
                         "Check epochs, and that seq-length exceeds the longest "
                         "episode.]")
        self.ended = time.time()
        self._append(f"[exit {self.returncode}]")

    def stop(self) -> None:
        if self.proc.poll() is None:
            # start_new_session put the child in its own process group, so this
            # also reaches anything it spawned (train.py -> dataloader workers).
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                self.proc.terminate()
            self.status = "stopping"

    def snapshot(self, offset: int = 0) -> dict:
        with self._lock:
            lines = self.lines[offset:]
            total = len(self.lines)
        return {
            "id": self.id, "kind": self.kind, "status": self.status,
            "returncode": self.returncode, "lines": lines, "next_offset": total,
            "elapsed": round((self.ended or time.time()) - self.started, 1),
            "losses": self.losses[-400:],
        }


JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()


def launch(kind: str, cmd: list[str], extra_env: dict | None = None) -> Job:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = os.pathsep.join([str(REPO), str(ICRT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    env.setdefault("WANDB_MODE", "disabled")
    if extra_env:
        env.update(extra_env)
    job = Job(kind, cmd, REPO, env)
    with JOBS_LOCK:
        JOBS[job.id] = job
    return job


# --------------------------------------------------------------------------- #
#  discovery
# --------------------------------------------------------------------------- #
def _read_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def describe_dataset(d: Path) -> dict:
    # dataset_config.json records num_arms / proprio_extra / action_extra since
    # the 23/21 change. Their ABSENCE is the signal that a set was converted at
    # the old 20-D layout and cannot load against a 23/21 model.
    cfg = _read_json(d / "dataset_config.json") or {}
    def _width(v):
        return len(v) if isinstance(v, (list, tuple)) else v
    dims = {k: _width(cfg[k]) for k in ("num_arms", "proprio_extra", "action_extra")
            if k in cfg}
    dim_names = {k: cfg[k] for k in ("proprio_extra", "action_extra")
                 if isinstance(cfg.get(k), (list, tuple))}
    lens = _read_json(d / "epi_len_mapping.json") or {}
    verbs = _read_json(d / "verb_to_episode.json") or {}
    episodes = sorted(lens.keys())
    values = [v for v in lens.values() if isinstance(v, (int, float))]
    tasks = [{"task": k, "n": len(v), "thin": len(v) < MIN_DEMOS}
             for k, v in sorted(verbs.items())] if isinstance(verbs, dict) else []
    # episode -> task, so the Evaluate tab can tell the operator which of the
    # three prompting conditions they are actually constructing. Without it,
    # "same task" vs "different task" is something you have to hold in your head.
    ep_task = {}
    if isinstance(verbs, dict):
        for task, eps in verbs.items():
            for e in (eps or []):
                ep_task[e] = task
    viz = d / "viz"
    longest = int(max(values)) if values else None
    return {
        "ep_task": ep_task or None,
        "dims": dims or None,
        "dim_names": dim_names or None,
        "legacy_dims": not dims,
        "seq_length_min": _next_pow2(longest) if longest else MIN_SEQ_LENGTH,
        "name": d.name,
        "path": str(d.relative_to(REPO)) if d.is_relative_to(REPO) else str(d),
        "episodes": episodes,
        "n_episodes": len(episodes),
        "len_min": min(values) if values else None,
        "len_max": max(values) if values else None,
        "tasks": tasks,
        "thin_tasks": [t["task"] for t in tasks if t["thin"]],
        "viz": sorted(p.stem for p in viz.glob("*.png")) if viz.is_dir() else [],
    }


_INTERP: dict | None = None


def interpreter_check() -> dict:
    """Probe the interpreter that jobs will actually run under.

    Jobs are spawned with sys.executable, so if the UI is started with a python
    that lacks the deps, EVERY job dies with ModuleNotFoundError and the log
    looks like a broken pipeline rather than a wrong interpreter. Check once and
    say so plainly.
    """
    global _INTERP
    if _INTERP is None:
        probe = ("import json,importlib;"
                 "m={};"
                 "\nfor n in ('numpy','torch','cv2','h5py','timm'):\n"
                 "    try:\n        importlib.import_module(n); m[n]=True\n"
                 "    except Exception:\n        m[n]=False\n"
                 "try:\n    import torch; m['cuda']=bool(torch.cuda.is_available())\n"
                 "except Exception:\n    m['cuda']=False\n"
                 "print(json.dumps(m))")
        try:
            out = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                                 text=True, timeout=180)
            mods = json.loads(out.stdout.strip().splitlines()[-1])
        except Exception:
            mods = {}
        # What each module actually gates, so a missing one names the buttons it
        # breaks rather than just a package. numpy/torch alone were not enough:
        # a venv with both still failed every visualize job on h5py.
        needed = {"numpy": "everything", "h5py": "convert, visualize, train, evaluate",
                  "cv2": "convert, visualize", "torch": "train, evaluate",
                  "timm": "train, evaluate"}
        missing = [k for k in needed if not mods.get(k)]
        _INTERP = {"path": sys.executable, "modules": mods,
                   "missing": missing,
                   "breaks": {k: needed[k] for k in missing},
                   "ok": not missing}
    return _INTERP


_TORCH = None


def _torch_mem():
    """Unified-memory total/used via torch, because nvidia-smi cannot report it
    on GB10.

    GB10 is a unified-memory part: `nvidia-smi --query-gpu=memory.used` returns
    literal "[N/A]". torch.cuda.mem_get_info() does work there. Imported lazily
    and once — the server already spawns torch subprocesses, so the cost is a
    one-off import, and mem_get_info per request is microseconds after that.
    """
    global _TORCH
    if _TORCH is None:
        try:
            import torch  # noqa: PLC0415
            _TORCH = torch if torch.cuda.is_available() else False
        except Exception:
            _TORCH = False
    if not _TORCH:
        return None, None
    try:
        free, total = _TORCH.cuda.mem_get_info()
        return (total - free) // 2 ** 20, total // 2 ** 20
    except Exception:
        return None, None


def _torch_peak():
    """Peak allocated by THIS process. Zero in the UI server (it never runs a
    model); the useful peak is the training job's own, which train.py logs."""
    if not _TORCH:
        return None
    try:
        return int(_TORCH.cuda.max_memory_allocated()) // 2 ** 20
    except Exception:
        return None


def _num(x):
    """nvidia-smi prints '[N/A]' for fields a part does not support. One of
    those must not take the whole readout down with it — which is exactly what
    an int() over the whole row did."""
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def gpu_state() -> dict:
    """Live GPU state for the header.

    The instruction was to make GPU use VISIBLE rather than inferred: an
    operator starting a training run should see the device is busy, not deduce
    it from how long an iteration takes.
    """
    rows = []
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,compute_cap,driver_version,memory.used,memory.total,"
             "utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        for line in out.stdout.strip().splitlines():
            f = [x.strip() for x in line.split(",")]
            if len(f) < 7:
                continue
            # Memory comes from TORCH on BOTH platforms, not from nvidia-smi.
            # GB10 is unified-memory and reports "[N/A]"; the RTX PRO 6000 has
            # discrete VRAM and would report a real number. Taking whichever
            # happens to work would mean the two machines show different
            # quantities under the same label. torch.mem_get_info is device-wide
            # and identical in meaning on both.
            used, total = _torch_mem()
            if used is None or total is None:       # no torch: fall back
                used, total = _num(f[3]), _num(f[4])
            rows.append({"name": f[0], "compute_cap": f[1], "driver": f[2],
                         "mem_used": used, "mem_total": total,
                         "mem_source": "torch" if _TORCH else "nvidia-smi",
                         "util": _num(f[5]), "temp": _num(f[6])})
    except Exception as e:
        return {"available": False, "gpus": [], "error": str(e)}
    return {"available": bool(rows), "gpus": rows}


def scan_state() -> dict:
    datasets, checkpoints = [], []
    data_root = REPO / "data"
    if data_root.is_dir():
        for d in sorted(data_root.iterdir()):
            if d.is_dir() and (d / "dataset_config.json").is_file():
                datasets.append(describe_dataset(d))
    # lerobot sources: a dir with meta/ but no dataset_config.json
    sources = []
    if data_root.is_dir():
        for d in sorted(data_root.iterdir()):
            if d.is_dir() and (d / "meta").is_dir() and not (d / "dataset_config.json").is_file():
                sources.append(str(d.relative_to(REPO)))
    for out in sorted(REPO.glob("output*/**/checkpoint*.pth")):
        yamls = sorted(out.parent.glob("*.yaml"))
        checkpoints.append({
            "path": str(out.relative_to(REPO)),
            "yaml": str(yamls[0].relative_to(REPO)) if yamls else "",
            "mtime": int(out.stat().st_mtime),
        })
    with JOBS_LOCK:
        jobs = [{"id": j.id, "kind": j.kind, "status": j.status,
                 "elapsed": round((j.ended or time.time()) - j.started, 1)}
                for j in sorted(JOBS.values(), key=lambda x: -x.started)]
    return {"interpreter": interpreter_check(), "gpu": gpu_state(),
            "datasets": datasets, "sources": sources,
            "checkpoints": sorted(checkpoints, key=lambda c: -c["mtime"]),
            "jobs": jobs, "repo": str(REPO),
            "limits": {"min_demos": MIN_DEMOS, "min_seq_length": MIN_SEQ_LENGTH}}


def _pos_int(body: dict, key: str, default: int, minimum: int = 1) -> int:
    """Read a positive integer from a request body.

    `body.get(key, default)` is NOT enough and this cost a real training run:
    the form always SENDS the key, so a cleared field arrives as 0 (JS `+""`)
    and the default never applies. The job then ran for 0 epochs, exited 0, and
    reported "done" having learned nothing.
    """
    raw = body.get(key, default)
    try:
        v = int(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{key}: expected a number, got {raw!r}")
    if v < minimum:
        raise ValueError(
            f"{key} must be at least {minimum}, got {v}"
            + (" — a 0-epoch run exits cleanly having trained nothing"
               if key == "epochs" else ""))
    return v


def _next_pow2(n: int) -> int:
    p = 1
    while p <= n:
        p *= 2
    return p


def train_warnings(dataset: Path, seq_length: int, maximum_length: int,
                   proprio_extra: int = 3, action_extra: int = 1) -> list[str]:
    """The three silent-failure modes from EXECUTION.md §4, checked against the
    dataset actually selected rather than left to the operator to remember."""
    out = []
    lens = _read_json(dataset / "epi_len_mapping.json") or {}
    values = [v for v in lens.values() if isinstance(v, (int, float))]
    longest = int(max(values)) if values else None
    if longest is not None and seq_length <= longest:
        out.append(
            f"seq-length {seq_length} does not exceed the longest episode "
            f"({longest} frames). With --no-prompt-loss most windows will contain no "
            f"episode boundary, loss reads a flat 0.0000, and training reports success "
            f"while learning nothing. Use at least {_next_pow2(longest)}.")
    elif longest is None and seq_length < MIN_SEQ_LENGTH:
        out.append(
            f"seq-length {seq_length} is below {MIN_SEQ_LENGTH} and this dataset has no "
            f"epi_len_mapping.json to check against. seq-length must exceed the longest "
            f"episode in frames.")
    if values and maximum_length <= max(values):
        out.append(
            f"maximum-length {maximum_length} does not exceed the longest episode "
            f"({int(max(values))} frames). Episodes at or above it are DROPPED — set "
            f"at least {int(max(values)) + 50}.")
    cfg = _read_json(dataset / "dataset_config.json") or {}
    if not any(k in cfg for k in ("num_arms", "proprio_extra", "action_extra")):
        out.append(
            f"this dataset predates the 23/21 action space — dataset_config.json "
            f"declares no num_arms/proprio_extra/action_extra, so it holds 20-D "
            f"proprio and action. Training it against the current defaults "
            f"(proprio_extra={proprio_extra}, action_extra={action_extra}) will fail "
            f"on a shape mismatch in the dataloader. Re-convert it.")
    else:
        # The converter records these as LISTS OF JOINT NAMES
        # (e.g. ["head_joint1","head_joint2","lift_joint"]), while the training
        # flags take a WIDTH. Compare the count, not the raw value — an int
        # comparison here fires a false mismatch on a correctly converted set.
        want = {"proprio_extra": proprio_extra, "action_extra": action_extra}
        for k, v in want.items():
            if k not in cfg:
                continue
            got = len(cfg[k]) if isinstance(cfg[k], (list, tuple)) else cfg[k]
            if got != v:
                names = f" ({', '.join(cfg[k])})" if isinstance(cfg[k], (list, tuple)) else ""
                out.append(
                    f"{k} mismatch: the dataset has {got}{names}, the form asks for {v}. "
                    f"The model would be built at the wrong width.")
    verbs = _read_json(dataset / "verb_to_episode.json") or {}
    thin = [f"{k} ({len(v)})" for k, v in verbs.items() if len(v) < MIN_DEMOS] \
        if isinstance(verbs, dict) else []
    if thin:
        out.append(
            f"tasks with fewer than {MIN_DEMOS} episodes will vanish in the "
            f"train/val split: {', '.join(thin)}")
    return out


# --------------------------------------------------------------------------- #
#  command builders
# --------------------------------------------------------------------------- #
def cmd_convert(b: dict) -> list[str]:
    c = [sys.executable, "-m", "aiworker_icrt.convert_lerobot_icrt",
         "--lerobot-root", b["lerobot_root"], "--out-dir", b["out_dir"],
         "--target-fps", str(b.get("target_fps", 15))]
    if b.get("image_size"):
        c += ["--image-size"] + str(b["image_size"]).split()
    if b.get("limit"):
        c += ["--limit", str(b["limit"])]
    if b.get("prompt_dir"):
        c += ["--prompt-dir", b["prompt_dir"]]
    for kv in (b.get("camera_map") or "").split():
        c += ["--camera-map", kv]
    return c


def cmd_visualize(b: dict) -> list[str]:
    c = [sys.executable, "-m", "aiworker_icrt.visualize",
         "--dataset", b["dataset"], "--episode", b["episode"],
         "--fps", str(b.get("fps", 15))]
    if b.get("limit"):
        c += ["--limit", str(b["limit"])]
    return c


def cmd_train(b: dict) -> list[str]:
    ds = REPO / b["dataset"]
    cmd = [sys.executable, str(ICRT / "scripts" / "train.py"),
            "--dataset-cfg.dataset-json", str(ds / "dataset_config.json"),
            "--dataset-cfg.maximum-length", str(_pos_int(b, "maximum_length", 1200)),
            "--dataset-cfg.non-overlapping", str(_pos_int(b, "non_overlapping", 256)),
            "--shared-cfg.num-arms", "2",
            # Absent, the model builds at the OLD 20-D width and dies on a shape
            # mismatch against 23/21 data. Defaulted here AND in the form.
            "--shared-cfg.proprio-extra-dim", str(b.get("proprio_extra", 3)),
            "--shared-cfg.action-extra-dim", str(b.get("action_extra", 1)),
            "--shared-cfg.num-cameras", "3",
            "--shared-cfg.seq-length", str(_pos_int(b, "seq_length", 512)),
            "--shared-cfg.batch-size", str(_pos_int(b, "batch_size", 1)),
            "--shared-cfg.num-pred-steps", "16",
            "--model-cfg.policy-cfg.scratch-llama-config",
            str(ICRT / "config" / "model_config" / "custom_transformer.json"),
            "--model-cfg.policy-cfg.phase", "pretrain",
            "--model-cfg.policy-cfg.no-prompt-loss",
            "--trainer-cfg.epochs", str(_pos_int(b, "epochs", 1)),
            "--trainer-cfg.accum-iter", str(_pos_int(b, "accum_iter", 2)),
            "--logging-cfg.output-dir", b.get("output_dir", "output"),
            "--logging-cfg.log-name", b.get("log_name", "icrt")]
    # rebalance_tasks resamples every task to a common length. Pointless on one
    # task and it silently reshapes a real multi-task set, so it is opt-IN here
    # rather than a default nobody sees.
    if b.get("rebalance_tasks"):
        cmd.append("--dataset-cfg.rebalance-tasks")
    return cmd


def cmd_evaluate(b: dict) -> list[str]:
    c = [sys.executable, "-m", "aiworker_icrt.evaluate",
         "--dataset", b["dataset"], "--checkpoint", b["checkpoint"],
         "--train-yaml", b["train_yaml"],
         "--prompt", b["prompt"], "--eval", b["eval"]]
    if b.get("max_steps"):
        c += ["--max-steps", str(b["max_steps"])]
    return c


BUILDERS = {"convert": cmd_convert, "visualize": cmd_visualize,
            "train": cmd_train, "evaluate": cmd_evaluate}


# --------------------------------------------------------------------------- #
#  http
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    server_version = "aiworker-ui"
    # HTTP/1.1 keeps the connection alive between range requests. A <video>
    # element issues many of them; on 1.0 each one costs a fresh TCP handshake.
    # Every response below sets Content-Length, which 1.1 requires.
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quieter than the default
        if "/api/job/" not in self.path:
            sys.stderr.write("  %s\n" % (fmt % args))

    # -- helpers ------------------------------------------------------------ #
    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def _serve_file(self, p: Path):
        """Serve a file with byte-range support.

        <video> needs this. Without Accept-Ranges the browser cannot seek at
        all, and Safari refuses to start playback outright — the element just
        sits there, which reads as "the video is broken" rather than "the server
        only does whole-file GETs".
        """
        size = p.stat().st_size
        ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        rng = self.headers.get("Range")
        if not rng:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(size))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with p.open("rb") as fh:
                while chunk := fh.read(256 * 1024):
                    self.wfile.write(chunk)
            return

        m = re.match(r"bytes=(\d*)-(\d*)$", rng.strip())
        if not m:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        start_s, end_s = m.groups()
        if start_s:
            start = int(start_s)
            end = int(end_s) if end_s else size - 1
        else:                                   # suffix form: bytes=-N
            start = max(0, size - int(end_s or 0))
            end = size - 1
        start, end = max(0, start), min(end, size - 1)
        if start > end:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        self.send_response(206)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        remaining = end - start + 1
        with p.open("rb") as fh:
            fh.seek(start)
            while remaining > 0:
                chunk = fh.read(min(256 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def _safe(self, rel: str) -> Path | None:
        """Resolve a client-supplied path inside the repo, or refuse."""
        p = (REPO / rel).resolve()
        return p if p.is_relative_to(REPO) and p.exists() else None

    # -- routes ------------------------------------------------------------- #
    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        if u.path == "/api/state":
            return self._json(scan_state())
        if u.path.startswith("/api/job/"):
            job = JOBS.get(u.path.rsplit("/", 1)[-1])
            if not job:
                return self._json({"error": "no such job"}, 404)
            return self._json(job.snapshot(int((q.get("offset") or ["0"])[0])))
        if u.path == "/api/file":
            p = self._safe((q.get("path") or [""])[0])
            if not p or not p.is_file():
                return self._json({"error": "not found"}, 404)
            return self._serve_file(p)
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        u = urlparse(self.path)
        try:
            body = self._body()
        except Exception as e:
            return self._json({"error": f"bad json: {e}"}, 400)

        if u.path.startswith("/api/job/") and u.path.endswith("/stop"):
            job = JOBS.get(u.path.split("/")[3])
            if not job:
                return self._json({"error": "no such job"}, 404)
            job.stop()
            return self._json({"ok": True})

        kind = u.path.rsplit("/", 1)[-1]
        if kind not in BUILDERS:
            return self._json({"error": "not found"}, 404)

        if kind == "train":
            ds = self._safe(body.get("dataset", ""))
            if ds is None:
                return self._json({"error": "dataset not found"}, 400)
            warns = train_warnings(ds, int(body.get("seq_length", 512)),
                                   int(body.get("maximum_length", 1200)),
                                   int(body.get("proprio_extra", 3)),
                                   int(body.get("action_extra", 1)))
            # Refuse the first time, list what is wrong, let the operator decide.
            # These three all "succeed" — refusing once is the only chance to
            # catch them before the run reports a clean finish having learnt
            # nothing.
            if warns and not body.get("confirm"):
                return self._json({"warnings": warns}, 409)
        try:
            job = launch(kind, BUILDERS[kind](body))
        except KeyError as e:
            return self._json({"error": f"missing field {e}"}, 400)
        except ValueError as e:
            return self._json({"error": str(e)}, 400)
        except Exception as e:
            return self._json({"error": str(e)}, 500)
        return self._json({"id": job.id})


def _say(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--host", default="127.0.0.1",
                    help="loopback by default; pass 0.0.0.0 deliberately to expose "
                         "a UI that can start GPU jobs on the robot network")
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    _say(f"aiworker-icrt UI  ->  http://{a.host}:{a.port}")
    _say(f"  repo   {REPO}")
    info = interpreter_check()
    _say(f"  python {info['path']}")
    if not info["ok"]:
        _say("  ** this interpreter is missing modules the jobs need:")
        for k, v in info["breaks"].items():
            _say(f"       {k:<7} breaks: {v}")
        _say("     Jobs are spawned with this interpreter, so those will fail with")
        _say("     ModuleNotFoundError. The container has all of them:")
        _say("       ./container.sh start")
        _say("       ./container.sh exec \"python -m aiworker_icrt.ui --host 0.0.0.0\"")
    else:
        mods = info["modules"]
        print(f"  deps   torch={mods.get('torch')} cuda={mods.get('cuda')} "
              f"cv2={mods.get('cv2')} h5py={mods.get('h5py')}")
    _say(f"  ctrl-c to stop")
    if a.host not in ("127.0.0.1", "localhost"):
        _say(f"  ** bound to {a.host} — reachable from the robot network **")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        _say("\nstopping; killing running jobs")
        for j in list(JOBS.values()):
            j.stop()
