from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import torch
import torchaudio
from jiwer import wer as jiwer_wer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import WhisperModel, WhisperProcessor

from .constants import ACCENTS
from .io import read_jsonl


def _load_mono(path: str | Path, sr: int = 16000) -> torch.Tensor:
    wav, orig_sr = torchaudio.load(str(path))
    wav = wav.mean(dim=0, keepdim=True)
    if orig_sr != sr:
        wav = torchaudio.functional.resample(wav, orig_sr, sr)
    return wav


def duration_seconds(path: str | Path) -> float:
    info = torchaudio.info(str(path))
    return float(info.num_frames) / float(info.sample_rate)


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class WhisperMeanExtractor:
    """Whisper-small encoder with valid-frame mean pooling (the curation scorer frontend)."""

    def __init__(self, model_id: str = "openai/whisper-small", device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperModel.from_pretrained(model_id).to(self.device).eval()
        self.model_id = model_id

    @torch.no_grad()
    def __call__(self, wav_path: str | Path) -> np.ndarray:
        wav = _load_mono(wav_path, 16000)[0]
        samples = int(wav.numel())
        inputs = self.processor(
            wav.numpy(),
            sampling_rate=16000,
            return_tensors="pt",
        )
        feats = inputs.input_features.to(self.device)
        hidden = self.model.encoder(input_features=feats).last_hidden_state[0]
        # Whisper: 10 ms mel hop followed by 2x time downsampling -> about 50 Hz.
        valid_frames = max(1, min(hidden.shape[0], math.ceil(samples / 320)))
        pooled = hidden[:valid_frames].mean(dim=0)
        return pooled.float().cpu().numpy()


class AccentScorerQ:
    def __init__(self, extractor: WhisperMeanExtractor, scaler, classifier):
        self.extractor = extractor
        self.scaler = scaler
        self.classifier = classifier
        self.classes = [str(x) for x in classifier.classes_]

    @classmethod
    def load(cls, path: str | Path, device: str = "cuda"):
        obj = joblib.load(path)
        extractor = WhisperMeanExtractor(obj["model_id"], device=device)
        return cls(extractor, obj["scaler"], obj["classifier"])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model_id": self.extractor.model_id,
                "scaler": self.scaler,
                "classifier": self.classifier,
            },
            path,
        )

    def probabilities(self, wav_path: str | Path) -> dict[str, float]:
        feat = self.extractor(wav_path)[None]
        x = self.scaler.transform(feat)
        p = self.classifier.predict_proba(x)[0]
        return {str(c): float(v) for c, v in zip(self.classifier.classes_, p)}


def train_q(
    train_manifest: str | Path,
    output_path: str | Path,
    *,
    model_id: str = "openai/whisper-small",
    device: str = "cuda",
) -> AccentScorerQ:
    extractor = WhisperMeanExtractor(model_id=model_id, device=device)
    X, y = [], []
    for row in read_jsonl(train_manifest):
        label = str(row["accent_label"])
        if label not in ACCENTS:
            raise ValueError(f"Unexpected accent label {label}")
        wav = row.get("audio_path", row.get("wav_path"))
        X.append(extractor(wav))
        y.append(label)
    X = np.stack(X)
    y = np.asarray(y)
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=3000,
    ).fit(scaler.transform(X), y)
    q = AccentScorerQ(extractor, scaler, clf)
    q.save(output_path)
    return q


class WhisperSmallASR:
    def __init__(self, device: str = "cuda", download_root: str | None = None):
        try:
            import whisper
        except ImportError as exc:
            raise RuntimeError("Install openai-whisper for WER scoring") from exc
        self.model = whisper.load_model("small", device=device, download_root=download_root)

    def transcribe(self, wav_path: str | Path) -> str:
        result = self.model.transcribe(str(wav_path), language="en", fp16=torch.cuda.is_available())
        return str(result["text"]).strip()


class EcapaSpeakerEncoder:
    def __init__(self, savedir: str | Path, device: str = "cuda"):
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            from speechbrain.pretrained import EncoderClassifier  # type: ignore
        run_opts = {"device": device if torch.cuda.is_available() else "cpu"}
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(savedir),
            run_opts=run_opts,
        )

    @torch.no_grad()
    def embedding(self, wav_path: str | Path) -> torch.Tensor:
        wav = _load_mono(wav_path, 16000)
        emb = self.model.encode_batch(wav).squeeze()
        return F_normalize(emb)


def F_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(p=2).clamp_min(1e-12)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(F_normalize(a.flatten()), F_normalize(b.flatten())).cpu())


def score_candidate(
    row: dict,
    q: AccentScorerQ,
    asr: WhisperSmallASR,
    spk: EcapaSpeakerEncoder,
) -> dict:
    out = dict(row)
    source_wav = out["source_wav"]
    output_wav = out["output_wav"]
    target = str(out["target_accent"])
    source_text = str(out["source_text"])

    source_probs = q.probabilities(source_wav)
    cand_probs = q.probabilities(output_wav)
    if target not in cand_probs:
        raise ValueError(f"Target accent {target} is absent from q classes")

    asr_text = asr.transcribe(output_wav)
    src_emb = spk.embedding(source_wav)
    out_emb = spk.embedding(output_wav)

    src_dur = duration_seconds(source_wav)
    out_dur = duration_seconds(output_wav)
    out.update(
        {
            "asr_text": asr_text,
            "wer": float(jiwer_wer(normalize_text(source_text), normalize_text(asr_text))),
            "speaker_sim": cosine(src_emb, out_emb),
            "duration_ratio": out_dur / max(src_dur, 1e-8),
            "target_prob": float(cand_probs[target]),
            "target_prob_delta": float(cand_probs[target] - source_probs[target]),
            "target_hit": int(max(cand_probs, key=cand_probs.get) == target),
        }
    )
    return out
