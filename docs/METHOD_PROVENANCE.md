# ICASSP method provenance

This repository is intentionally scoped to the **ICASSP 2027 accent-conversion system only**.
It contains the method, curation, training, and inference path needed for that submission; external baselines and the independent final evaluator remain separate dependencies.

## 1. Final method chain

The recovered production chain is:

1. Initialize the Vevo2 AR model from the official pretrained Vevo2 checkpoint.
2. Build an **accent-adapted categorical generator** `G_phi` using earlier reference-conditioned selected pseudo-target token trajectories.
3. Freeze `G_phi`.
4. For each final `(source transcript t, target accent a)` condition, generate **N=8** content-style token trajectories by stochastic decoding. No target reference waveform or target-reference transcript is used in this final search.
5. Decode each trajectory through frozen Vevo2 Flow Matching + Vocos while using the **source waveform as the timbre/speaker reference**.
6. Score each waveform for accent, content, speaker similarity, and duration.
7. Apply the fixed feasibility gate, then take the lexicographic **Top-1** trajectory.
8. Train the final categorical accent converter on the resulting synthetic token supervision using teacher-forced autoregressive CE/NLL.
9. At inference, use the source transcript + categorical target accent in the AR model, then the source waveform only for frozen acoustic/timbre synthesis. Inference is single-pass (`best-of-n=1`).

## 2. Fixed experiment facts

### Vevo2 provenance

- Amphion commit: `26f6883110181f1dbfe95c70a7c7dbaf4de5f42a`
- Base AR checkpoint root: `Amphion/ckpts/Vevo2/contentstyle_modeling/posttrained`
- Fixed candidate-generator checkpoint in the experiment tree: `scales/30k/training/af_ce_r32_qkvo/epoch_8/advanced_adapter.pt`
- `G_phi` adaptation population: 29,840 earlier selected pseudo-target trajectories.
- Final candidate-construction population: 29,754 source-utterance/target-accent conditions.
- Final candidate bank size: 29,754 x 8 = 238,032 trajectories.

### Categorical AR model

- `prompt_length=32`
- target accent string -> `ACCENT_TO_ID` -> learned continuous `AccentPromptBank`
- target accent is **not tokenized as text**
- shared LoRA target modules: `q_proj,k_proj,v_proj,o_proj`
- shared LoRA rank: 32
- shared LoRA alpha: 64
- LoRA dropout: 0.05
- accent-specific LoRA rank: 0
- max sequence length: 1024
- phone prefix: disabled
- phone cross-attention: disabled
- deep FiLM: disabled
- auxiliary accent/content/preference losses: disabled
- main objective: autoregressive token CE/NLL

### Final N=8 generation

- `do_sample=True`
- `top_k=25`
- `top_p=0.8`
- `temperature=1.0`
- `min_new_tokens=15`
- `max_new_tokens=500`
- Flow Matching steps: 32
- base seed: 1337
- attempt seed: `1337 + stable_int(f"{base_pair_id}::attempt={attempt}")`
- source-prosody conditioning: disabled

### Candidate feasibility gate

A candidate is eligible iff:

- WER <= 0.08
- ECAPA speaker similarity >= 0.60
- duration ratio in [0.75, 1.45]
- target accent hit = 1 **OR** target-accent probability delta >= 0.02

Among eligible candidates, Top-1 is selected lexicographically by:

`(target_prob, target_prob_delta, target_hit, speaker_sim, -WER)`.

Final HardTop1 retained set: **16,153 conditions**.

### Candidate accent scorer q

The curation scorer used in the recovered experiment implementation is:

- Whisper-small encoder
- valid-frame mean pooling
- `StandardScaler`
- balanced `LogisticRegression`
- six classes: Arabic, Chinese, Hindi, Korean, Spanish, Vietnamese
- `C=1.0`
- solver `lbfgs`
- `max_iter=3000`
- recovered held-out dev size: 1,200 (200/accent)
- recovered dev accuracy: 86.33%
- recovered dev Macro-F1: 86.18%

This scorer is **not** the final frozen four-fold paper evaluator.

### Final converter training

Recovered final HardTop1 schedule:

- epochs: 3
- optimizer: AdamW
- shared QKVO LoRA LR: `5e-6`
- micro-batch: 8
- gradient accumulation: 6
- weight decay: 0.01
- warmup ratio: 0.05
- bf16 autocast
- seeds: 1337, 2027, 3407
- backbone frozen

Historical audit records seed-specific selected epochs E1 / E2 / E2 for 1337 / 2027 / 3407.

## 3. Intentionally not fabricated

The project-chat audit did **not** recover all of the following with enough confidence to hard-code them as historical facts:

1. the original file identity/path of the 6,000-row training manifest used for `q`;
2. a separate learning rate for `AccentPromptBank` during final HardTop1 training;
3. exact optimizer/LR/micro-batch/gradient-accumulation metadata for the **earlier 29,840-trajectory adaptation run** that produced `G_phi`;
4. the exact original implementation lines of `_batch_build` / label masking in the legacy `AdvancedAccentStudent`.

The release implementation therefore makes these explicit/configurable rather than inventing undocumented values.
Before making a byte-identical checkpoint-training reproduction claim, compare these unrecovered pieces against the original experiment metadata.
