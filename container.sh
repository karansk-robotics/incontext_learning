#!/usr/bin/env bash
#
# container.sh — lifecycle wrapper for the ICRT + ROS 2 Jazzy (Zenoh) stack.
#
#   ./container.sh build [--data] [--control]   build image (--data: LeRobot,
#                                   --control: pinocchio/osqp for cyclo_control)
#   ./container.sh build-control    colcon-build cyclo_control's solver core
#   ./container.sh parity           FK vs Pinocchio check (isolated numpy<2 venv)
#   ./container.sh start [--router] bring up icrt (+ the network's single zenohd)
#   ./container.sh enter [svc]      interactive shell (default: icrt)
#   ./container.sh stop             stop and remove the containers
#   ./container.sh restart          stop then start
#   ./container.sh status           container state + GPU/RMW/zenoh sanity check
#   ./container.sh logs [svc]       follow logs (default: icrt)
#   ./container.sh exec <cmd...>    run one command inside icrt
#
# ZENOH: exactly ONE rmw_zenohd per robot network.
#   router host:  ./container.sh start --router
#   other hosts:  ZENOH_ROUTER_ENDPOINT=tcp/<router-ip>:7447 ./container.sh start
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# --- zenoh ------------------------------------------------------------------
# Shape borrowed from ROBOTIS' own cyclo_intelligence stack, which drives the
# same robot over rmw_zenoh. Three things adopted from theirs:
#   1. IP and PORT as separate knobs (they default 127.0.0.1:7447 and branch on
#      whether IP is still loopback, same single-router logic as ours)
#   2. shared-memory transport ON. We move three camera streams; SHM keeps
#      same-host traffic out of the TCP loopback path. Needs ipc:host, which
#      docker-compose.yml already sets.
#   3. mode="client" when the router is REMOTE. The rmw_zenoh session default is
#      "peer"; ROBOTIS switches to client for the remote-router case.
export ZENOH_ROUTER_IP="${ZENOH_ROUTER_IP:-127.0.0.1}"
export ZENOH_ROUTER_PORT="${ZENOH_ROUTER_PORT:-7447}"
export ZENOH_SHM_ENABLED="${ZENOH_SHM_ENABLED:-true}"
# Back-compat: an explicit endpoint still wins over IP/PORT.
export ZENOH_ROUTER_ENDPOINT="${ZENOH_ROUTER_ENDPOINT:-tcp/${ZENOH_ROUTER_IP}:${ZENOH_ROUTER_PORT}}"

_zenoh_is_local() {
    case "$ZENOH_ROUTER_ENDPOINT" in
        *localhost*|*127.0.0.1*|*\[::1\]*) return 0 ;;
        *) return 1 ;;
    esac
}

if [ -z "${ZENOH_CONFIG_OVERRIDE:-}" ]; then
    # Multiple pairs are ';'-separated; each is path=json5value, applied via
    # zenoh's zc_config_insert_json5.
    _ov="transport/shared_memory/enabled=${ZENOH_SHM_ENABLED}"
    _ov="${_ov};connect/endpoints=[\"${ZENOH_ROUTER_ENDPOINT}\"]"
    _zenoh_is_local || _ov="${_ov};mode=\"client\""
    export ZENOH_CONFIG_OVERRIDE="$_ov"
fi

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"

# ------------------------------------------------------------------ platform
case "$(uname -m)" in
    aarch64|arm64)
        export ICRT_DOCKERFILE="docker/DockerFile.arm64"
        ARCH_LABEL="aarch64 SBSA (DGX Spark GB10 / Jetson Thor)" ;;
    x86_64|amd64)
        export ICRT_DOCKERFILE="docker/DockerFile.x86"
        ARCH_LABEL="x86_64 (Blackwell)" ;;
    *)
        echo "container.sh: unsupported architecture: $(uname -m)" >&2; exit 1 ;;
esac

# ---------------------------------------------------------------- docker cli
# Three cases, in order of preference:
#   1. docker works directly
#   2. the user IS in the docker group but this login session predates the
#      membership, so its process credentials are stale. `id -nG` omits docker
#      while `getent group docker` lists the user. sg re-execs with the group
#      applied — no sudo, no password. This is the common case on this box.
#   3. genuinely no access -> sudo
if docker info >/dev/null 2>&1; then
    DC=(docker compose); DOCKER=(docker)
elif [ -z "${ICRT_SG_REEXEC:-}" ] && sg docker -c 'docker info' >/dev/null 2>&1; then
    export ICRT_SG_REEXEC=1
    exec sg docker -c "$(printf '%q ' "$0" "$@")"
else
    if [ -n "${ICRT_SG_REEXEC:-}" ]; then
        echo "container.sh: re-exec under 'sg docker' still cannot reach the daemon." >&2
        exit 1
    fi
    if ! sudo -n true 2>/dev/null; then
        echo "note: cannot reach the docker daemon and 'sg docker' did not help."
        if getent group docker | grep -q "\b${USER}\b"; then
            echo "      you ARE in the docker group — log out and back in (or: newgrp docker)."
        else
            echo "      to fix: sudo usermod -aG docker \$USER && newgrp docker"
        fi
    fi
    DC=(sudo -E docker compose); DOCKER=(sudo docker)
fi

log()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*"; }

cmd="${1:-}"; shift || true

case "$cmd" in
    build)
        BUILD_ARGS=()
        while [[ "${1:-}" == --* ]]; do
            case "$1" in
                --data)    export INSTALL_DATA_DEPS=1
                           log "including LeRobot conversion stack" ;;
                --control) export INSTALL_CONTROL_DEPS=1
                           log "including cyclo_control build deps (pinocchio, osqp)" ;;
                *) echo "unknown build flag: $1" >&2; exit 1 ;;
            esac
            shift
        done
        log "building for ${ARCH_LABEL} using ${ICRT_DOCKERFILE}"
        "${DC[@]}" build "${BUILD_ARGS[@]}" "$@"
        ;;

    start)
        PROFILES=()
        RUN_ROUTER=0
        if [[ "${1:-}" == "--router" ]]; then
            shift; RUN_ROUTER=1; PROFILES=(--profile router)
        fi

        log "starting on ${ARCH_LABEL}"
        if [ -n "${DISPLAY:-}" ] && command -v xhost >/dev/null 2>&1; then
            xhost +local:root >/dev/null 2>&1 || true
        fi

        "${DC[@]}" "${PROFILES[@]}" up -d "$@"
        echo
        "${DC[@]}" "${PROFILES[@]}" ps

        echo
        if [ "$RUN_ROUTER" = 1 ]; then
            log "this host runs THE zenoh router (listening tcp/[::]:7447)"
            log "point other hosts at it:  ZENOH_ROUTER_ENDPOINT=tcp/$(hostname -I 2>/dev/null | awk '{print $1}'):7447 ./container.sh start"
        else
            log "zenoh client -> ${ZENOH_ROUTER_ENDPOINT}"
            if [[ "$ZENOH_ROUTER_ENDPOINT" == *localhost* || "$ZENOH_ROUTER_ENDPOINT" == *127.0.0.1* ]]; then
                warn "no router started and the endpoint is local — ROS discovery will"
                warn "not work until something runs rmw_zenohd. Either:"
                warn "    ./container.sh start --router          (make this host the router)"
                warn "    ZENOH_ROUTER_IP=<router-ip> …        (point at the router host)"
            fi
        fi
        ;;

    build-control)
        # cyclo_control's solver core. Source is bind-mounted, the build is ~25 s
        # and 1.8 MB, so it is built into the repo rather than baked into the
        # image -- that keeps one copy and avoids an image rebuild when the
        # controller source changes. Needs an image built with
        # --build-arg INSTALL_CONTROL_DEPS=1.
        log "building cyclo_motion_controller_core in the container"
        "${DC[@]}" exec icrt bash -lc '
            set -e
            command -v pinocchio-config >/dev/null 2>&1 || \
              [ -d /opt/ros/$ROS_DISTRO/include/pinocchio ] || {
                echo "pinocchio headers missing - rebuild the image with" >&2
                echo "  ./container.sh build --control" >&2
                exit 1; }
            cd /workspace/third_party/cyclo_control
            colcon build --packages-select osqp_eigen_vendor cyclo_motion_controller_core \
                --cmake-args -DCMAKE_BUILD_TYPE=Release -Wno-dev' ;;

    parity)
        # The FK-vs-Pinocchio check runs in its own numpy<2 venv — see the
        # Dockerfile for why it cannot share the main interpreter.
        log "FK vs Pinocchio parity (sidecar venv)"
        "${DC[@]}" exec icrt bash -lc '
            [ -x /opt/pinvenv/bin/python ] || {
              echo "parity sidecar missing — rebuild with --build-arg INSTALL_PARITY=1" >&2
              exit 1; }
            cd /workspace
            PYTHONPATH=/workspace:/workspace/third_party/icrt \
              /opt/pinvenv/bin/python -m pytest tests/ -k pinocchio -q -rs' ;;

    enter)
        "${DC[@]}" exec "${1:-icrt}" bash ;;

    exec)
        [ $# -gt 0 ] || { echo "usage: container.sh exec <command...>" >&2; exit 1; }
        "${DC[@]}" exec icrt bash -lc "$*" ;;

    stop)
        log "stopping"
        "${DC[@]}" --profile router down "$@" ;;

    restart)
        "$0" stop; "$0" start "$@" ;;

    logs)
        "${DC[@]}" --profile router logs -f "${1:-icrt}" ;;

    status)
        "${DC[@]}" --profile router ps
        echo
        log "host GPU"
        nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv 2>/dev/null \
            || echo "  nvidia-smi unavailable"
        # Show the host driver against the one the image was BUILT against, so a
        # mismatch reads as a host problem rather than a broken image. A driver
        # older than the image's build driver fails at CUDA init with an error
        # that looks nothing like "your driver is too old".
        HOSTDRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
        IMGDRV=$("${DOCKER[@]}" image inspect "aiworker-icrt:${ICRT_TAG:-latest}" \
                   --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null \
                 | grep '^CUDA_DRIVER_VERSION=' | cut -d= -f2)
        if [ -n "$HOSTDRV" ] && [ -n "$IMGDRV" ]; then
            echo "  host driver  $HOSTDRV"
            echo "  image built against  $IMGDRV"
            if [ "$(printf '%s\n%s\n' "$IMGDRV" "$HOSTDRV" | sort -V | head -1)" != "$IMGDRV" ]; then
                warn "host driver is OLDER than the image's build driver — CUDA init may"
                warn "fail. Either update the driver, or build with an older BASE_IMAGE:"
                warn "  25.08-py3 -> 580.65.06   25.09-py3 -> 580.82.07   25.11-py3 -> 580.95.05"
            fi
        fi
        echo
        log "zenoh"
        echo "  endpoint = ${ZENOH_ROUTER_ENDPOINT}"
        echo "  shm      = ${ZENOH_SHM_ENABLED}"
        echo "  override = ${ZENOH_CONFIG_OVERRIDE}"
        if "${DOCKER[@]}" ps --format '{{.Names}}' 2>/dev/null | grep -qx icrt-zenohd; then
            echo "  router                = running on THIS host"
        else
            echo "  router                = not on this host (expected if another host runs it)"
        fi
        echo
        if "${DOCKER[@]}" ps --format '{{.Names}}' 2>/dev/null | grep -qx icrt; then
            log "in-container"
            "${DC[@]}" exec -T icrt bash -lc '
                python -c "import torch;print(f\"  torch     {torch.__version__} / cuda {torch.version.cuda}\");print(f\"  gpu       {torch.cuda.is_available()} {torch.cuda.get_device_capability() if torch.cuda.is_available() else \"\"}\")"
                echo "  rmw       ${RMW_IMPLEMENTATION}"
                python -c "import icrt" 2>/dev/null && echo "  icrt      importable" || echo "  icrt      NOT importable"
                python -c "import aiworker_icrt" 2>/dev/null && echo "  aiworker  importable" || echo "  aiworker  NOT importable"
                python -c "import scipy,yaml" 2>/dev/null && echo "  scipy+yaml present" || echo "  scipy/yaml MISSING"
                python -c "import cv2;print(f\"  cv2       {cv2.__version__}\")" 2>/dev/null || echo "  cv2       MISSING"
                # Distinguish "ROS env missing" from "ROS fine, no router". These
                # used to collapse into one message, which masked a real bug:
                # exec ran without the ROS environment and status still looked healthy.
                if ! command -v ros2 >/dev/null 2>&1; then
                    echo "  ros2      *** NOT ON PATH — ROS env not sourced (BUG, not a router issue)"
                elif ! python -c "import rclpy" 2>/dev/null; then
                    echo "  ros2      *** on PATH but rclpy will not import (BUG)"
                elif timeout 5 ros2 node list >/dev/null 2>&1; then
                    echo "  ros2      responding"
                else
                    echo "  ros2      env OK, no nodes (expected until a router runs)"
                fi
            ' || true
        else
            echo "  (icrt not running — ./container.sh start)"
        fi
        ;;

    ""|-h|--help|help)
        sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//' ;;

    *)
        echo "container.sh: unknown command '$cmd' (try --help)" >&2; exit 1 ;;
esac
