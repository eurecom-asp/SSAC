#!/usr/bin/env bash
set -euo pipefail

# Required environment variables:
#   SSAC_ROOT          this repository
#   AMPHION_ROOT       Amphion checkout at 26f6883110181f1dbfe95c70a7c7dbaf4de5f42a
#   VEVO2_ROOT         usually $AMPHION_ROOT/ckpts/Vevo2
#   GPHI_CKPT          released/recovered fixed generator checkpoint
#   Q_MODEL            fitted curation scorer q (.joblib)
#   CONDITIONS         final 29,754-row condition manifest
#   ECAPA_DIR          local SpeechBrain ECAPA cache
# Optional:
#   WARMSTART_1337 / WARMSTART_2027 / WARMSTART_3407

: "${SSAC_ROOT:?}"
: "${AMPHION_ROOT:?}"
: "${VEVO2_ROOT:?}"
: "${GPHI_CKPT:?}"
: "${Q_MODEL:?}"
: "${CONDITIONS:?}"
: "${ECAPA_DIR:?}"

export PYTHONPATH="$SSAC_ROOT:$AMPHION_ROOT:${PYTHONPATH:-}"
AR_CKPT="$VEVO2_ROOT/contentstyle_modeling/posttrained"
RUN="$SSAC_ROOT/runs/icassp_release"
mkdir -p "$RUN"

python "$SSAC_ROOT/cli/generate_candidates.py" \
  --conditions "$CONDITIONS" \
  --ar-checkpoint "$AR_CKPT" \
  --student-checkpoint "$GPHI_CKPT" \
  --vevo2-root "$VEVO2_ROOT" \
  --output-dir "$RUN/candidates_wav" \
  --results-jsonl "$RUN/candidates.jsonl" \
  --best-of-n 8 \
  --flow-steps 32 \
  --top-k 25 --top-p 0.8 --temperature 1.0 \
  --min-new-tokens 15 --max-new-tokens 500 \
  --seed 1337

python "$SSAC_ROOT/cli/score_candidates.py" \
  --candidates "$RUN/candidates.jsonl" \
  --q "$Q_MODEL" \
  --ecapa-dir "$ECAPA_DIR" \
  --output "$RUN/scores.jsonl"

python "$SSAC_ROOT/cli/select_top1.py" \
  --scores "$RUN/scores.jsonl" \
  --output "$RUN/hard_top1.jsonl"

for SEED in 1337 2027 3407; do
  INIT_VAR="WARMSTART_${SEED}"
  INIT="${!INIT_VAR:-}"
  EXTRA=()
  if [[ -n "$INIT" ]]; then EXTRA+=(--init-checkpoint "$INIT"); fi
  python "$SSAC_ROOT/cli/train_student.py" \
    --config "$SSAC_ROOT/configs/final_hardtop1.yaml" \
    --manifest "$RUN/hard_top1.jsonl" \
    --ar-checkpoint "$AR_CKPT" \
    --output-dir "$RUN/final_seed${SEED}" \
    --seed "$SEED" \
    "${EXTRA[@]}"
done

python "$SSAC_ROOT/cli/audit_manifests.py" \
  --conditions "$CONDITIONS" \
  --candidates "$RUN/candidates.jsonl" \
  --scores "$RUN/scores.jsonl" \
  --selected "$RUN/hard_top1.jsonl"
