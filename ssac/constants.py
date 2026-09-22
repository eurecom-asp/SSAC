from __future__ import annotations

ACCENTS = ("Arabic", "Chinese", "Hindi", "Korean", "Spanish", "Vietnamese")
ACCENT_TO_ID = {name: idx for idx, name in enumerate(ACCENTS)}

# Final ICASSP candidate-feasibility gate recovered from the project audit.
MAX_WER = 0.08
MIN_SPEAKER_SIM = 0.60
MIN_DURATION_RATIO = 0.75
MAX_DURATION_RATIO = 1.45
MIN_TARGET_PROB_DELTA = 0.02

# Final stochastic decoding setup.
DEFAULT_TOP_K = 25
DEFAULT_TOP_P = 0.8
DEFAULT_TEMPERATURE = 1.0
DEFAULT_MIN_NEW_TOKENS = 15
DEFAULT_MAX_NEW_TOKENS = 500
DEFAULT_FLOW_STEPS = 32
DEFAULT_BASE_SEED = 1337
DEFAULT_CANDIDATES = 8
