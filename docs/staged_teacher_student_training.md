# Staged teacher--student training

This repository provides a standalone LinVel + asynchronous-depth teacher--student
baseline.  It is intentionally separate from the existing `RepTS` tasks.

| Stage | Task / mode | Rollout controller | Trainable modules |
| --- | --- | --- | --- |
| 1 | `Mjlab-Velocity-Rough-WF-Tron1B-TS-Teacher` | privileged teacher | privileged encoder, actor, critic, latent-dynamics predictor |
| 2 | `Mjlab-Velocity-Rough-WF-Tron1B-TS-LinVel-Depth` (`warm_start`) | teacher | depth/proprio encoder, latent head, LinVel head |
| 3 | same task (`dagger`) | student | student actor plus encoder (separate learning rates) |

Stage 1 feeds the teacher actor the noisy proprio history used by the student,
the clean command, a noisy ground-truth base linear velocity (`Uniform[-0.02,
0.02]`), and the explicit privileged latent.  The actor never receives the full
critic observation directly.  The critic and privileged encoder remain clean.

## Commands

To run all three stages with the default 30,000 iterations per stage, use:

```bash
bash scripts/rsl_rl/train_staged_teacher_student.sh --gpu-ids [0]
```

The script locates each final checkpoint, wires it into the next stage, and
creates a fresh optimizer at the warm-start → DAgger boundary.  For a short
smoke test or a different budget, set `TEACHER_ITERATIONS`,
`WARM_START_ITERATIONS`, and `DAGGER_ITERATIONS` before the command.  Any
trailing arguments are applied to all three stages.

Train the privileged teacher first:

```bash
uv run python scripts/rsl_rl/train.py \
  Mjlab-Velocity-Rough-WF-Tron1B-TS-Teacher
```

Then start the encoder warm-start with the resulting teacher checkpoint.  The
runner copies and freezes the teacher actor head after loading the checkpoint.

```bash
uv run python scripts/rsl_rl/train.py \
  Mjlab-Velocity-Rough-WF-Tron1B-TS-LinVel-Depth \
  --agent.teacher-checkpoint /absolute/path/to/teacher/model_29999.pt
```

Warm-start keeps the teacher frozen and uses teacher-controlled rollouts.  Its
loss is

\[
L = \lVert z_S-z_T\rVert^2 + \lVert \hat v_S-v\rVert^2
  + 0.2\operatorname{Huber}(\mu_S,\mu_T).
\]

For stage 3, resume a warm-start checkpoint and switch the training stage.  The
stage-aware loader restores student and teacher weights but intentionally starts
a fresh optimizer: warm-start has one optimizer group, while DAgger has separate
actor and encoder groups.

```bash
uv run python scripts/rsl_rl/train.py \
  Mjlab-Velocity-Rough-WF-Tron1B-TS-LinVel-Depth \
  --agent.teacher-checkpoint /absolute/path/to/teacher/model_29999.pt \
  --agent.resume True \
  --agent.load-run <warm-start-run-directory> \
  --agent.load-checkpoint "model_29999.pt" \
  --agent.algorithm.training-stage dagger
```

Stage 3 is pure DAgger: every rollout action comes from the student immediately;
there is no teacher/student action mixture.  The frozen teacher labels those
student states online.  Its objective retains latent and velocity alignment and
adds policy KL:

\[
L = \lVert z_S-z_T\rVert^2 + \lVert \hat v_S-v\rVert^2
  + \operatorname{KL}(\pi_T\,\Vert\,\pi_S)
  + 0.2\operatorname{Huber}(\mu_S,\mu_T).
\]

The default actor learning rate is `1e-3`; the encoder learning rate is `2e-4`.
The depth GRU is optimized with truncated BPTT over 12 environment steps.  The
exported student ONNX policy takes `proprio_history`, `actor_command`, `depth`,
and `hidden_state_in`, and returns actions, predicted LinVel, and
`hidden_state_out`.
