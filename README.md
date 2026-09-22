<div align="center">

# SSAC

### Multi-Constraint Synthetic Supervision for Direct Accent Conversion

<p>
  <a href="https://github.com/eurecom-asp/SSAC"><img src="https://img.shields.io/badge/Code-GitHub-181717?logo=github&logoColor=white" alt="GitHub"></a>
  <a href="https://yangyangqu.github.io/accent-conversion-demo/"><img src="https://img.shields.io/badge/Audio-Demo-8A2BE2?logo=githubpages&logoColor=white" alt="Audio Demo"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/Backbone-Vevo2-5C4EE5" alt="Vevo2">
</p>

[**Audio Demo**](https://yangyangqu.github.io/accent-conversion-demo/) · [**Repository**](https://github.com/eurecom-asp/SSAC)

</div>

---

## Overview

SSAC converts speech toward a **categorical target accent** while preserving linguistic content and source-speaker characteristics.

The key idea is to use a fixed accent-conditioned generator to produce multiple candidate trajectories for each training condition, evaluate them under complementary accent, content, speaker, and duration constraints, and retain a single high-quality candidate as synthetic supervision for a final converter.

The multi-candidate stage is used **only for supervision construction**.
At inference time, the final converter runs in a **single pass** from source transcript and target-accent label, while the source waveform is used only by the frozen acoustic/timbre synthesis stage.

### Highlights

- **Categorical accent control** for six target accents: Arabic, Chinese, Hindi, Korean, Spanish, and Vietnamese.
- **Multi-candidate synthetic supervision** with `N=8` candidates per training condition.
- **Constraint-aware curation** using accent strength, intelligibility, speaker similarity, and duration consistency.
- **Hard Top-1 selection** to convert multiple stochastic candidates into one supervision target.
- **Parameter-efficient adaptation** with a 32-token accent prompt and shared Q/K/V/O LoRA.
- **Single-pass deployment**: best-of-N search is not required at inference time.
- Built on the open-source **Vevo2 / Amphion** generation stack.

---

## Method

### Training-time supervision construction


### Inference


A useful implementation detail is that the **source waveform is not used as an AR conditioning reference** for target-accent generation.
It is used only in the frozen acoustic/timbre stage to preserve source-speaker characteristics.

---

## Core configuration

| Component | Setting |
|---|---|
| Target accents | Arabic, Chinese, Hindi, Korean, Spanish, Vietnamese |
| Accent prompt | 32 continuous embeddings |
| LoRA modules | shared `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| LoRA rank / alpha | 32 / 64 |
| Candidate budget | `N=8` |
| WER threshold | `<= 0.08` |
| Speaker similarity | `>= 0.60` |
| Duration ratio | `[0.75, 1.45]` |
| Accent feasibility | target hit **or** `Δp_target >= 0.02` |
| Top-1 ranking | `(p_target, Δp_target, hit, SIM, -WER)` |
| Final objective | autoregressive token CE/NLL |

The full configuration is stored in [`configs/`](configs/).

---

## Installation

### 1. Clone SSAC

```bash
git clone git@github.com:eurecom-asp/SSAC.git
cd SSAC
```

### 2. Set up Vevo2 / Amphion

SSAC uses the Vevo2 implementation from Amphion.
The experiments were developed against the following Amphion revision:

```text
26f6883110181f1dbfe95c70a7c7dbaf4de5f42a
```

```bash
git clone https://github.com/open-mmlab/Amphion.git
cd Amphion
git checkout 26f6883110181f1dbfe95c70a7c7dbaf4de5f42a
pip install -r models/svc/vevo2/requirements.txt
```

### 3. Install SSAC

```bash
cd /path/to/SSAC
pip install -e .

export SSAC_ROOT=/path/to/SSAC
export AMPHION_ROOT=/path/to/Amphion
export VEVO2_ROOT=$AMPHION_ROOT/ckpts/Vevo2
export PYTHONPATH=$SSAC_ROOT:$AMPHION_ROOT:$PYTHONPATH
```

Vevo2 checkpoints should follow the original Amphion directory layout under `ckpts/Vevo2`.

---

## Training pipeline

### 1. Train the candidate accent scorer

The curation scorer uses a Whisper-small encoder with valid-frame mean pooling followed by a `StandardScaler` and balanced logistic regression.

```bash
python cli/train_q.py \
  --train-manifest data/q_train.jsonl \
  --dev-manifest data/q_dev.jsonl \
  --output ckpts/q_whisper_small_lr.joblib
```

### 2. Generate the candidate bank

```bash
python cli/generate_candidates.py \
  --conditions data/conditions.jsonl \
  --ar-checkpoint "$VEVO2_ROOT/contentstyle_modeling/posttrained" \
  --student-checkpoint /path/to/G_phi \
  --vevo2-root "$VEVO2_ROOT" \
  --output-dir runs/n8/wav \
  --results-jsonl runs/n8/candidates.jsonl \
  --best-of-n 8 \
  --flow-steps 32 \
  --top-k 25 \
  --top-p 0.8 \
  --temperature 1.0 \
  --min-new-tokens 15 \
  --max-new-tokens 500 \
  --seed 1337
```

For a fixed `(source_text, target_accent)` condition, the candidates share the same model conditioning and differ only through stochastic decoding.

### 3. Score candidates

```bash
python cli/score_candidates.py \
  --candidates runs/n8/candidates.jsonl \
  --q ckpts/q_whisper_small_lr.joblib \
  --ecapa-dir ckpts/speechbrain_ecapa \
  --output runs/n8/scores.jsonl
```

### 4. Apply the feasibility gate and Hard Top-1 selection

```bash
python cli/select_top1.py \
  --scores runs/n8/scores.jsonl \
  --output runs/n8/hard_top1.jsonl
```

The selector first removes infeasible candidates and then applies the accent-first lexicographic ranking defined in [`configs/selection.yaml`](configs/selection.yaml).

### 5. Train the final converter

```bash
python cli/train_student.py \
  --config configs/final_hardtop1.yaml \
  --manifest runs/n8/hard_top1.jsonl \
  --ar-checkpoint "$VEVO2_ROOT/contentstyle_modeling/posttrained" \
  --output-dir runs/final \
  --seed 1337
```

For multi-run experiments, repeat training with the desired random seeds and retain checkpoints according to the development protocol used for your experiment.

---

## Data and manifest format

SSAC expects JSONL manifests for training conditions, generated candidates, and selected supervision.
Detailed field definitions and validation rules are documented in:

- [`docs/MANIFESTS.md`](docs/MANIFESTS.md)
- [`docs/METHOD_PROVENANCE.md`](docs/METHOD_PROVENANCE.md)

You can audit a manifest before running expensive generation or training:

```bash
python cli/audit_manifests.py --help
```

---

## Repository structure

```text
SSAC/
├── ssac/                    # Core model, generation, scoring, and selection code
│   ├── student.py
│   ├── generation.py
│   ├── scoring.py
│   ├── select.py
│   ├── training.py
│   └── vevo2_backend.py
│
├── cli/                     # Command-line entry points
│   ├── adapt_generator.py
│   ├── generate_candidates.py
│   ├── score_candidates.py
│   ├── select_top1.py
│   ├── train_q.py
│   ├── train_student.py
│   ├── infer.py
│   └── audit_manifests.py
│
├── configs/                 # Generation, selection, and training configurations
├── docs/                    # Manifest and provenance documentation
├── tests/                   # Lightweight regression tests
├── requirements.txt
└── pyproject.toml
```

---

## Reproducibility notes

This repository separates **verified implementation details** from historical metadata that could not be reliably reconstructed.
Unrecovered details are documented explicitly rather than replaced with guessed values.

For the cleanest reproduction workflow:

1. keep the Amphion revision fixed;
2. use the exact condition manifest associated with the experiment;
3. keep candidate-generation seeds deterministic;
4. run manifest audits before generation and training;
5. keep the candidate scorer separate from the final evaluation model.

See [`docs/METHOD_PROVENANCE.md`](docs/METHOD_PROVENANCE.md) for the full provenance boundary.

---

## Audio examples

Listening examples across target accents and comparison systems are available here:

**https://yangyangqu.github.io/accent-conversion-demo/**

---

## Acknowledgements

SSAC builds on [Vevo2](https://github.com/open-mmlab/Amphion/tree/main/models/svc/vevo2) and the [Amphion](https://github.com/open-mmlab/Amphion) toolkit.
We thank the authors and maintainers of these open-source projects.

