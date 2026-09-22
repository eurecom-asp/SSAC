# SSAC — ICASSP 2027 Accent Conversion

Clean public release implementation for the ICASSP accent-conversion system developed on top of Vevo2.
This repository contains only the ICASSP method path: categorical accent conditioning, multi-candidate synthetic supervision, feasibility filtering/Top-1 selection, and final converter training/inference.

## What the method does

The method turns reference-based accent transfer into categorical target-accent conversion.
A fixed accent-adapted Vevo2-based generator `G_phi` produces multiple possible content-style token trajectories for the same source transcript and target-accent label.
The trajectories are synthesized with the source waveform as the timbre reference, screened for accent/content/speaker/duration constraints, and the strongest feasible trajectory becomes synthetic supervision for a final converter.

Final inference is **not best-of-N**: the final converter takes source transcript + target-accent label, samples one content-style trajectory, and uses the source waveform only in the frozen Vevo2 acoustic/timbre stage.

## Frozen experiment design

- target accents: Arabic / Chinese / Hindi / Korean / Spanish / Vietnamese
- categorical accent prompt: 32 continuous embeddings
- Vevo2 AR adaptation: shared Q/K/V/O LoRA, rank 32, alpha 64
- accent-specific LoRA: disabled (`rank=0`)
- source-prosody conditioning: disabled
- candidate bank: `N=8`
- final construction: 29,754 conditions -> 238,032 candidate trajectories
- historical condition manifest: `historical_balanced_full_29754.jsonl` = 17,649 LibriTTS + 12,105 L2-ARCTIC rows, 4,959 per target accent; same-accent L2-ARCTIC pairs excluded
- fixed gate: WER <= 0.08, SIM >= 0.60, duration ratio in [0.75, 1.45], and target-hit OR delta target probability >= 0.02
- final selection: lexicographic `(target_prob, delta, hit, SIM, -WER)`
- HardTop1 retained training conditions: 16,153
- final converter objective: token autoregressive CE/NLL
- final training: 3 epochs, AdamW, shared-LoRA LR `5e-6`, micro-batch 8, grad-accum 6, WD 0.01, 5% warmup, bf16; seeds 1337/2027/3407

See [`docs/METHOD_PROVENANCE.md`](docs/METHOD_PROVENANCE.md) for the evidence boundary and the few historical details that were not recovered rather than guessed.

## Dependency: Vevo2 / Amphion

Experiments used Amphion commit:

```text
26f6883110181f1dbfe95c70a7c7dbaf4de5f42a
```

Set up Amphion separately and keep its repository root on `PYTHONPATH`:

```bash
git clone https://github.com/open-mmlab/Amphion.git
cd Amphion
git checkout 26f6883110181f1dbfe95c70a7c7dbaf4de5f42a
pip install -r models/svc/vevo2/requirements.txt
```

Then install SSAC:

```bash
cd /path/to/SSAC
pip install -e .
export PYTHONPATH=/path/to/SSAC:/path/to/Amphion:$PYTHONPATH
```

Vevo2 checkpoints follow the original Amphion layout under `ckpts/Vevo2`.

## 1. Candidate scorer q

The curation scorer is Whisper-small encoder -> valid-frame mean pooling -> StandardScaler -> balanced LogisticRegression.

```bash
python cli/train_q.py \
  --train-manifest data/q_train.jsonl \
  --dev-manifest data/q_dev.jsonl \
  --output ckpts/q_whisper_small_lr.joblib
```

The recovered held-out protocol reports `N=1200`, accuracy `86.33%`, Macro-F1 `86.18%`.
The exact historical 6000-row training-manifest file was not recovered from the project audit, so it is not fabricated here.

## 2. Fixed generator G_phi

For exact paper reproduction, use the recovered fixed generator checkpoint corresponding to:

```text
scales/30k/training/af_ce_r32_qkvo/epoch_8/advanced_adapter.pt
```

The loader accepts either the checkpoint directory or the `advanced_adapter.pt` file directly.

It was created from 29,840 earlier selected reference-conditioned pseudo-target trajectories, then chosen by waveform-level development evaluation.
The release also contains `cli/adapt_generator.py` so this stage is implemented, but its historical optimizer/LR/batch metadata must be filled from the original run metadata because those values were not fully recovered from the chats.

## 3. Generate the N=8 final bank

Use the historical `historical_balanced_full_29754.jsonl` as `conditions.jsonl`; it contains one row per source-utterance/target-accent condition.
The original per-source target-assignment algorithm was not safely recovered, so the release validates the historical manifest instead of inventing a replacement; see [`docs/MANIFESTS.md`](docs/MANIFESTS.md).

```bash
python cli/generate_candidates.py \
  --conditions data/conditions.jsonl \
  --ar-checkpoint "$VEVO2_ROOT/contentstyle_modeling/posttrained" \
  --student-checkpoint ckpts/G_phi \
  --vevo2-root "$VEVO2_ROOT" \
  --output-dir runs/n8/wav \
  --results-jsonl runs/n8/candidates.jsonl \
  --best-of-n 8 \
  --flow-steps 32 \
  --top-k 25 --top-p 0.8 --temperature 1.0 \
  --min-new-tokens 15 --max-new-tokens 500 \
  --seed 1337
```

The eight attempts share exactly the same `(source_text, target_accent, G_phi)` conditioning; only the deterministic stochastic-decoding seed changes.
There is no target-reference waveform/transcript in this final N=8 construction.

## 4. Score and select HardTop1

```bash
python cli/score_candidates.py \
  --candidates runs/n8/candidates.jsonl \
  --q ckpts/q_whisper_small_lr.joblib \
  --ecapa-dir ckpts/speechbrain_ecapa \
  --output runs/n8/scores.jsonl

python cli/select_top1.py \
  --scores runs/n8/scores.jsonl \
  --output runs/n8/hard_top1.jsonl
```

The selector first applies the fixed gate, discards conditions with no feasible candidate, then performs the accent-first lexicographic ranking.

## 5. Train the final converter

Run all three paper seeds and keep all epoch checkpoints for the waveform-level development selection:

```bash
for seed in 1337 2027 3407; do
  python cli/train_student.py \
    --config configs/final_hardtop1.yaml \
    --manifest runs/n8/hard_top1.jsonl \
    --ar-checkpoint "$VEVO2_ROOT/contentstyle_modeling/posttrained" \
    --output-dir "runs/final_seed${seed}" \
    --seed "$seed"
done
```

The historical audit records selected epochs E1/E2/E2 for seeds 1337/2027/3407.
If the original `quality_gate_v2` seed-specific warm starts are being reproduced, pass them with `--init-checkpoint`.

## 6. Single-pass inference

```bash
python cli/infer.py \
  --ar-checkpoint "$VEVO2_ROOT/contentstyle_modeling/posttrained" \
  --student-checkpoint runs/final_seed1337/epoch_1 \
  --vevo2-root "$VEVO2_ROOT" \
  --source-wav example/source.wav \
  --source-text "The source transcript." \
  --target-accent Korean \
  --output-wav out.wav
```

## One-command paper pipeline

After setting the environment variables documented at the top of the script:

```bash
bash scripts/reproduce_icassp.sh
```

## Scope

This repository contains only the method, curation, training, and inference path used for the ICASSP accent-conversion submission.
External baseline implementations and the independent final paper evaluator remain separate dependencies.
