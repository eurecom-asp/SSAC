# Manifest schemas

All manifests are JSONL.

## `conditions.jsonl`

One row per final source-utterance/target-accent condition:

```json
{
  "pair_id": "src0001__to__Korean",
  "source_utt_id": "src0001",
  "source_wav": "/path/to/src0001.wav",
  "source_text": "The transcript of the source utterance.",
  "source_speaker": "speaker-id",
  "source_accent": "Arabic",
  "target_accent": "Korean"
}
```

The authoritative historical manifest is named `historical_balanced_full_29754.jsonl`.
The final audited construction contains 29,754 rows: 17,649 LibriTTS sources and 12,105 L2-ARCTIC sources.
Target accents are exactly Arabic, Chinese, Hindi, Korean, Spanish, and Vietnamese, with 4,959 conditions per target accent.
For L2-ARCTIC sources, same-accent source/target pairs are excluded.

The historical **assignment algorithm** that maps every source utterance to its one target accent was not recovered with enough confidence from the chats.
Therefore this release treats the historical 29,754-row manifest as an input artifact and validates it; it does not fabricate a new target-assignment algorithm that could silently change the paper population.

## `candidates.jsonl`

`cli/generate_candidates.py` adds:

- `base_pair_id`
- `candidate_id`
- `best_of_n_attempt`
- `candidate_seed`
- `output_wav`
- `content_style_ids`
- `num_content_style_codes`
- optional `error`

A complete N=8 bank has 238,032 rows.

## `scores.jsonl`

`cli/score_candidates.py` additionally adds:

- `asr_text`
- `wer`
- `speaker_sim`
- `duration_ratio`
- `target_prob`
- `target_prob_delta`
- `target_hit`

## `hard_top1.jsonl`

`cli/select_top1.py` keeps one eligible trajectory per retained condition and adds aliases expected by the trainer:

- `teacher_cs_ids`
- `positive_cs_ids`
- `selection_rule="hard_top1"`

The final audited HardTop1 set contains 16,153 rows.

## `q_train.jsonl` / `q_dev.jsonl`

The scorer trainer expects:

```json
{"audio_path": "/path/to.wav", "accent_label": "Chinese"}
```

The recovered original protocol used six balanced accents and a 1,200-row held-out development split (200/accent).
The exact historical training-manifest file identity was not recovered from the chats, so this repository does not invent it.
