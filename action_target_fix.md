# The action target was three different quantities

**The model was being trained to predict servo lag, and the servo lag was tuned
differently in each of our three recording sessions. Nothing in the camera image
says which session a frame came from, so a large part of the target was
unpredictable in principle — and MSE's optimal answer to an unpredictable target
is the average. That is exactly what the model emitted.**

Found 2026-09-14 while investigating why the best policy scored **worse than
outputting nothing at all** on a third of the held-out episodes.

---

## 1. The symptom

`mt21_warm/checkpoint-12`, the best policy we had at 0.0308 m, lost to the
trivial *"command the pose you already observe"* rule on 6 of 8 held-out
episodes. On two of them it lost by a factor of 55 and 65.

That is not a weak model. A weak model predicts small, roughly-correct motions.
A model that is 65x worse than emitting **zero** is answering a different
question from the one being scored.

## 2. Ruling out the code first

Both transforms in the delta path are exact:

```
scale_action -> unscale_action            max |x - roundtrip(x)|  8.754e-08
convert_delta_action -> convert_abs_action  position error        1.1e-16
```

Also verified: the statistics file used at training (`R dx std 0.03755`, the
correct 151-episode version) was written at 20:23 and unchanged at 20:32 when
training loaded it. No mid-flight replacement.

So the inconsistency was in the recordings, not the pipeline.

## 3. The cause

Teleoperation has two arms: the **leader** the human holds, and the **follower**
the robot runs. The follower trails the leader by the servo lag. We recorded both
— `action` is the leader, `observation.state` is the follower.

That lag is a property of how the controller was tuned on the day:

```
task    |action - state| (joint)   at the hand
box          0.006736 rad            21.1 mm
ecu          0.009399 rad            38.4 mm
ylw2         0.000223 rad             0.75 mm      <- 30-40x less
```

ICRT's regression target is `action[t+j] - proprio[t]`. With the leader source
that is **21-38 mm of lag in two tasks and 0.75 mm in the third**. The word
"action" meant *"3 cm ahead of the robot"* in `box`/`ecu` and *"where the robot
already is"* in `ylw2`.

**The model cannot see controller gains.** They are not in the images and not in
the proprio. Asked to predict a quantity whose scale is set by information it
does not receive, it did the only sensible thing: emit the average, a roughly
constant ~22 mm delta everywhere. Right for `box`, 65x too large for `ylw2`.

## 4. The fix

Take the target from the follower's **achieved pose one step ahead** instead of
the leader's command:

```
before   action[t] = joints_to_action( leader_command[t] )
after    action[t] = joints_to_action( follower_state[t+1] )
```

Real motion, visible in the video, meaning the same thing in every session and
independent of controller tuning. It is also what a policy should emit — the next
pose, leaving the servo to close the gap.

Measured over all 151 episodes:

```
working-arm |action - observation|
  before (leader command)   18.42 mm   range [ 0.54 .. 51.89]    96x spread
  after  (achieved pose)     8.70 mm   range [ 4.68 .. 14.00]     3x spread
```

### Two implementations, both in the repo

- `aiworker_icrt/convert_lerobot_icrt.py --action-source achieved` for new
  conversions.
- `aiworker_icrt/retarget_actions.py` rewrites an existing converted dataset in
  minutes instead of an hour of video decode. This is **exact, not an
  approximation**: `joints_to_action` and `joints_to_proprio` both call
  `joints_to_cartesian` and append the same lift channel
  (`ACTION_EXTRA == PROPRIO_EXTRA == [lift_joint]`), so the Cartesian action of a
  joint vector IS its Cartesian proprio. Shifting the stored proprio forward one
  frame is byte-identical to re-converting.

## 5. It worked — the pathology is gone

Per-episode ratio of predicted motion to true motion, working arm:

```
                          AFTER (ep2, fixed)   BEFORE (ep12, broken)
box_episode_000009              5.1x                  2.0x
box_episode_000021              5.3x                  0.8x
box_episode_000036              5.5x                  0.8x
box_episode_000047              5.4x                  1.0x
ecu_episode_000017              3.2x                  0.7x
ecu_episode_000037              2.6x                  0.6x
ylw2_episode_000006_pick        5.0x                 54.5x    <-
ylw2_episode_000013_pick        5.3x                 65.4x    <-
```

**`ylw2` is no longer an outlier.** And the truth column shows the label fix
directly: true motion is now 4.3-12.3 mm across all eight episodes, where before
it ranged 0.4 mm to 31 mm.

## 6. What is NOT fixed, and one correction

**The model over-predicts motion ~2.6-5.5x across every task**, and held-out MAE
is flat at ~0.042 m between epochs 2 and 4 against a **0.0074 m** bar. Early --
the previous run looked bad until epoch 8 -- but if the ratio is still ~5x at
epoch 12-16 there is a second problem underneath. The hypothesis to test then:
delta statistics are computed over 16-step chunks whose later elements are ~16x
larger than the first, so a model fit to that distribution may systematically
overshoot the one-step target we now score against.

**Correction to an earlier claim.** I said the 0.0065 m baseline was "inflated by
this defect" and would weaken. It did not: recomputed on the corrected targets it
is **0.0074 m**, essentially unchanged. The *repeat-the-previous-action* baseline
was always honest, because it measures one step of motion either way. Only the
*command-the-observed-pose* baseline was distorted (0.4 mm on `ylw2`), and it has
now converged onto the same value. **The bar stands where it stood.**

## 7. Why the ICRT authors never hit this

Their `dataset_config.json` uses `action/cartesian_position` — the leader command,
the same choice we originally made. It is not wrong; it is wrong *across
sessions*. All 1,550 ICRT-MT episodes came from one DROID platform in one lab
over a few weeks, so their lag is a constant of the world and the model absorbs
it. Ours came from three sessions with three tunings.

Their `preprocess_droid.py` also deletes every frame where
`movement_enabled == False`, which removes exactly the idle periods where leader
and follower converge — keeping their lag distribution tighter still. We keep
those frames.

**This matters beyond today's bug.** Merging the 857 BG2 episodes means a
different robot with certainly different controller tuning. On the leader target
that merge would reintroduce this problem at larger scale. On the achieved-pose
target it is safe.

## 8. Reproduce

```bash
python -m aiworker_icrt.retarget_actions \
    --src /dev/shm/icrt_multitask_split --out /dev/shm/icrt_mt_achieved
PYTHONPATH=third_party/icrt:. python make_stats.py \
    /dev/shm/icrt_mt_achieved /dev/shm/icrt_mt_achieved/action_stats_achieved_v1.json
./train_warm_ach.sh
PYTHONPATH=third_party/icrt:. python validate_achieved.py 4,8,12,16,20
```

`validate_achieved.py` recomputes both baselines on whatever targets it is given,
rather than carrying a hardcoded number forward — which is how the distorted
0.4 mm `ylw2` baseline survived as long as it did.
