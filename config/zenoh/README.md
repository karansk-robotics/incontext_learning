# Zenoh topology for the AI Worker stack

**Decision: exactly ONE `rmw_zenohd` for the whole robot network.** Not one per
host. Two routers on the same network both serve, and you get duplicated and
looped traffic.

## Why this is a decision at all

`rmw_zenoh_cpp` nodes are Zenoh *sessions* in `peer` mode that connect to a
router. With `network_mode: host` and the Spark, the Thor and the robot all on
the same LAN, the naive setup — a `zenohd` service per compose stack — gives you
one router per machine. Each is reachable by the others, so they federate and
every message can arrive twice.

## Layout

Pick one host to be the router. The robot's own compute is usually the right
choice (it is always on), but any host works.

```bash
# router host — runs rmw_zenohd + the policy
./container.sh start --router

# every other host — policy only, pointed at the router
ZENOH_ROUTER_ENDPOINT=tcp/192.168.1.50:7447 ./container.sh start
```

`./container.sh start` without `--router` starts no router at all (the `zenohd`
service sits behind a compose profile) and warns if the endpoint is still
`localhost`, which would mean nothing is routing.

## How the repointing works

Verified against `ros-jazzy-rmw-zenoh-cpp` 0.2.10, from the configs it ships in
`/opt/ros/jazzy/share/rmw_zenoh_cpp/config/`:

| | default | meaning |
|---|---|---|
| `DEFAULT_RMW_ZENOH_ROUTER_CONFIG.json5` | `mode: "router"`, `listen.endpoints: ["tcp/[::]:7447"]` | already listens on **all** interfaces — a single router is reachable network-wide with no edit |
| `DEFAULT_RMW_ZENOH_SESSION_CONFIG.json5` | `mode: "peer"`, `connect.endpoints: ["tcp/localhost:7447"]` | every node looks for a router on its **own** host |

So only the client side needs changing, and only one key. `docker-compose.yml`
sets it on the `icrt` service:

```yaml
ZENOH_CONFIG_OVERRIDE: 'connect/endpoints=["${ZENOH_ROUTER_ENDPOINT:-tcp/localhost:7447}"]'
```

`ZENOH_CONFIG_OVERRIDE` is applied through zenoh's `zc_config_insert_json5`
(confirmed by symbol in `librmw_zenoh_cpp.so`), i.e. one `path=json5value` pair.
The `zenohd` service deliberately does **not** get this variable — a router must
not be pointed at itself as a client.

## Relevant env vars

`rmw_zenoh_cpp` reads these (all confirmed present in the 0.2.10 binary):

| var | used by | note |
|---|---|---|
| `ZENOH_SESSION_CONFIG_URI` | nodes | full replacement session config |
| `ZENOH_ROUTER_CONFIG_URI` | `rmw_zenohd` | full replacement router config |
| `ZENOH_CONFIG_OVERRIDE` | both | single `path=value` patch; what we use |
| `ZENOH_ROUTER_CHECK_ATTEMPTS` | nodes | how long a node waits for a router before giving up |

If a future setup needs more than one key changed, drop a full session config
next to this file and set `ZENOH_SESSION_CONFIG_URI` instead of stacking
overrides.

## Sanity check

```bash
./container.sh status          # prints the endpoint and whether a router is local
./container.sh exec 'ros2 node list'
```

An empty `ros2 node list` with everything else healthy almost always means no
router is reachable — check `ZENOH_ROUTER_ENDPOINT` and that the router host is
actually up.

> Not yet validated on hardware: nothing in this stack has been built or run.
> Docker requires root on the Spark and the user is not in the `docker` group.

---

## Aligned with ROBOTIS' own stack (2026-09-14)

`~/robotis/cyclo_intelligence` drives this same robot over `rmw_zenoh`, so its
configuration is the closest thing to a reference implementation. Checked ours
against it.

**The mechanism matches.** They also configure Zenoh purely through
`ZENOH_CONFIG_OVERRIDE` (25 call sites) rather than replacement config files —
same `path=json5value` pairs, `;`-separated. No reason to switch approach.

**Three things adopted from theirs:**

1. **Shared-memory transport on** — `transport/shared_memory/enabled=true`.
   We move three camera streams; SHM keeps same-host traffic off the TCP
   loopback path. Requires `ipc: host`, which compose already sets.
2. **`mode="client"` when the router is remote.** The shipped session default is
   `mode: "peer"`; ROBOTIS switches to client for the remote-router case.
   `container.sh` now adds this automatically when the endpoint is not loopback.
3. **`ZENOH_ROUTER_IP` / `ZENOH_ROUTER_PORT` as separate knobs**, defaulting to
   `127.0.0.1:7447`, branching on whether IP is still loopback — the same
   single-router logic we arrived at independently.
   `ZENOH_ROUTER_ENDPOINT` still works and takes precedence.

Resulting override, composed by `container.sh`:

```
# local router
transport/shared_memory/enabled=true;connect/endpoints=["tcp/127.0.0.1:7447"]

# remote router
transport/shared_memory/enabled=true;connect/endpoints=["tcp/10.0.0.5:7447"];mode="client"
```

The `zenohd` service gets the SHM pair only — never `connect/endpoints` or
`mode`, which would point the router at itself.

**Not adopted: `zenoh_ros2_sdk` itself.** It is an uninitialised git submodule
(`ROBOTIS-GIT/zenoh_ros2_sdk`) and empty on this machine, so it could not be
read. From its surrounding code it is a message-definition / typesupport cache
layer — `docker/scripts/init_zenoh_cache.sh` pre-populates ROS interface
packages so containers skip a runtime git fetch — which is a different concern
from routing topology. Revisit if we ever need Zenoh-native pub/sub without a
ROS node; not needed for `rmw_zenoh`.
