# Recording spec — the same-scene task pair

**Goal: two tasks that are visually identical until you know the instruction.**

This is the one recording that makes the paper's central claim testable. Without
it, condition 3 has no interpretation — "the model ignores the prompt" and "the
prompt was unnecessary" predict exactly the same result, and we cannot tell them
apart. With it, the experiment either passes or fails on its merits.

Effort: about an afternoon. ~100 episodes of a ~9 second task is ~15 minutes of
robot time plus setup.

---

## 1. Why the current data cannot test it

Each of our three tasks was recorded in its own scene at its own head angle:

```
task    head1     head2      pairwise difference
box     0.466     0.348      box vs ecu    10.1 deg
ecu     0.643     0.348      box vs ylw2   18.2 / 31.4 deg
ylw2    0.784    -0.199      ecu vs ylw2    8.1 / 31.4 deg
```

Constant to 0.18 deg within a task. So the camera pose alone identifies the task.

Measured separability of **frame 0**, using nothing but mean RGB — the crudest
possible feature, before any motion, before any learning. The number is the
between-task difference divided by the within-task spread:

```
                      box vs ecu   box vs ylw2   ecu vs ylw2
cam_head                 19.6x         7.0x          2.6x
cam_wrist_left            3.4x         3.4x          4.2x
cam_wrist_right           0.1x         3.8x          3.1x
```

**Every camera leaks.** A network separates them far more easily than mean RGB
does. So during training the cheapest way to reduce loss was "recognise the
scene, emit that task's motion" — and reading the prompt bought nothing.
Gradient descent does not learn an ability it is never forced to use.

Measured consequence (`condition3_result.md`): prompting with a *different* task
costs 1.0 mm on a 55.6 mm error. Prompting with the exact episode about to be
scored does not help either.

Note `cam_wrist_right` scoring **0.1x** on box vs ecu — it is the idle arm in
both, parked looking at the same thing, carrying no task signal. That is what we
want *every* view to look like at the moment of decision.

## 2. The task pair

Two tasks whose scenes are **indistinguishable until the instruction is known**.

**Good.** Two objects of the same type, both present: *"pick the left one"* /
*"pick the right one"*. Or stacked: *"take the top"* / *"take the bottom"* —
which is exactly ICRT-MT's `context-top-yellow-cup` / `context-bottom-orange-cup`,
50 episodes each. The authors built this control into their training data
deliberately.

**Bad.** Two different objects in two different places. That is our current
dataset again, with new names.

## 3. Hard requirements

| # | requirement | what breaks without it |
|---|---|---|
| 1 | **Both targets present in every episode, in both tasks** | the scene reveals the answer |
| 2 | **One head angle, identical across both tasks** | camera pose becomes a task label (currently 8-31 deg apart) |
| 3 | **The same arm does the work in both** | "which arm moves" becomes the label — it already is in our data (ylw2 right, box/ecu left) |
| 4 | **Same table, lighting and object set** | whichever differs becomes the label instead |
| 5 | **~50 episodes per task** | matches ICRT-MT's context pair; below `min_demos = 4` a task is dropped entirely |
| 6 | **Short episodes, ~130 frames (~9 s at 15 fps)** | so a demonstration AND an attempt fit one `seq_length` window. ICRT-MT's median is 129; ours were 958, which is why we had to cut them at the grasp |
| 7 | **Same fps and controller settings for both** | fps changes the per-step delta scale; controller tuning changes the servo lag (see `action_target_fix.md`) |
| 8 | **Two distinct task strings in the LeRobot recording** | `verb_to_episode` groups on the task string; identical strings merge the tasks |

### Requirement 2 has an alternative

Randomise the head angle **per episode, independently of task** (say +/-15 deg).
That also breaks the correlation and is more robust — the model learns the task
does not depend on camera pose.

**If you do this, the head must go back into proprio** (the original 23-D layout
`[L arm 10 | R arm 10 | head1, head2, lift]`). A varying head with no signal
telling the model where it points means the same scene looks different for no
observable reason. One line in `constants.py` plus a re-conversion.

If you fix the head instead, the current 21-D layout stays correct.

## 4. Acceptance check, before training on it

Run the same separability measurement on the new recording:

```
frame-0 separability, mean RGB, between-task / within-task
  current data     19.6x on cam_head      <- trivially separable
  target           ~1x on every camera    <- indistinguishable
```

Anything much above ~1.5x means something still identifies the task, and it is
worth finding what before spending a training run.

Also confirm, per `validations.md` discipline:

- episode lengths in the 100-160 frame range
- both task strings present in `verb_to_episode.json` with ~50 episodes each
- head angle identical (or decorrelated) across the two
- one episode of each rendered with `visualize.py` and actually watched

## 5. What it then makes possible

Condition 3 becomes a real experiment: same observations, prompt from task A vs
task B. Same-task prompting wins -> in-context learning works on this robot.
It does not -> in-context learning genuinely fails here, and for the first time
that answer means something.

Either result is publishable. The current situation — an untestable claim — is
not.

## 6. What this does NOT fix

The policy's accuracy. Held-out error sits at 0.0309 m against a 0.0074 m bar
across two runs on completely different targets (`session_report.md`). That
ceiling is a separate problem: 151 episodes against 92.6 M parameters, and a
frozen vision encoder whose features correlate only ~0.3 with hand position.

More tasks and more episodes help that too. But this recording is about the
in-context claim specifically, and that is the claim the paper rests on.
