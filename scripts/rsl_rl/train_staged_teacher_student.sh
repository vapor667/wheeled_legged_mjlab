#!/usr/bin/env bash
# Train the independent teacher-student pipeline end to end.
#
# Optional environment variables:
#   RUN_ID=experiment_label
#   TEACHER_ITERATIONS=30000
#   WARM_START_ITERATIONS=30000
#   DAGGER_ITERATIONS=30000
# Remaining arguments are forwarded to every training stage, e.g.
#   bash scripts/rsl_rl/train_staged_teacher_student.sh --gpu-ids [0] --env.scene.num-envs 1024

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

run_id="${RUN_ID:-ts_$(date +%Y%m%d_%H%M%S)}"
teacher_iterations="${TEACHER_ITERATIONS:-30000}"
warm_start_iterations="${WARM_START_ITERATIONS:-10000}"
dagger_iterations="${DAGGER_ITERATIONS:-10000}"
common_args=("$@")

for iteration_count in "$teacher_iterations" "$warm_start_iterations" "$dagger_iterations"; do
  if ! [[ "$iteration_count" =~ ^[1-9][0-9]*$ ]]; then
    echo "Iteration counts must be positive integers, got: $iteration_count" >&2
    exit 2
  fi
done

teacher_task="Mjlab-Velocity-Rough-WF-Tron1B-TS-Teacher"
student_task="Mjlab-Velocity-Rough-WF-Tron1B-TS-LinVel-Depth"
teacher_experiment="wf_tron1b_ts_teacher"
student_experiment="wf_tron1b_ts_lin_vel_depth"

teacher_run_name="${run_id}_teacher"
warm_start_run_name="${run_id}_warm_start"
dagger_run_name="${run_id}_dagger"

echo "[TS] Stage 1/3: privileged teacher (${teacher_iterations} iterations)"
uv run python scripts/rsl_rl/train.py "$teacher_task" \
  "${common_args[@]}" \
  --agent.experiment-name "$teacher_experiment" \
  --agent.max-iterations "$teacher_iterations" \
  --agent.resume False \
  --agent.run-name "$teacher_run_name"

shopt -s nullglob
teacher_run_dirs=("logs/rsl_rl/${teacher_experiment}"/*_"${teacher_run_name}")
if (( ${#teacher_run_dirs[@]} != 1 )); then
  echo "Could not uniquely locate teacher run for ${teacher_run_name}." >&2
  exit 1
fi
teacher_checkpoint="${teacher_run_dirs[0]}/model_$((teacher_iterations - 1)).pt"
if [[ ! -f "$teacher_checkpoint" ]]; then
  echo "Teacher checkpoint was not written: $teacher_checkpoint" >&2
  exit 1
fi

echo "[TS] Stage 2/3: student encoder warm-start (${warm_start_iterations} iterations)"
uv run python scripts/rsl_rl/train.py "$student_task" \
  "${common_args[@]}" \
  --agent.experiment-name "$student_experiment" \
  --agent.max-iterations "$warm_start_iterations" \
  --agent.resume False \
  --agent.run-name "$warm_start_run_name" \
  --agent.algorithm.training-stage warm_start \
  --agent.teacher-checkpoint "$teacher_checkpoint"

warm_start_run_dirs=("logs/rsl_rl/${student_experiment}"/*_"${warm_start_run_name}")
if (( ${#warm_start_run_dirs[@]} != 1 )); then
  echo "Could not uniquely locate warm-start run for ${warm_start_run_name}." >&2
  exit 1
fi
warm_start_dir="${warm_start_run_dirs[0]}"
warm_start_checkpoint="${warm_start_dir}/model_$((warm_start_iterations - 1)).pt"
if [[ ! -f "$warm_start_checkpoint" ]]; then
  echo "Warm-start checkpoint was not written: $warm_start_checkpoint" >&2
  exit 1
fi

echo "[TS] Stage 3/3: pure student-rollout DAgger (${dagger_iterations} iterations)"
uv run python scripts/rsl_rl/train.py "$student_task" \
  "${common_args[@]}" \
  --agent.experiment-name "$student_experiment" \
  --agent.max-iterations "$dagger_iterations" \
  --agent.run-name "$dagger_run_name" \
  --agent.teacher-checkpoint "$teacher_checkpoint" \
  --agent.resume True \
  --agent.load-run "$(basename "$warm_start_dir")" \
  --agent.load-checkpoint "$(basename "$warm_start_checkpoint")" \
  --agent.algorithm.training-stage dagger

echo "[TS] Complete. Run id: ${run_id}"
