from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import torch


class Vevo2Decoder:
    """Frozen Vevo2 Flow-Matching + Vocos synthesis using the source waveform as timbre reference."""

    def __init__(self, vevo2_root: str | Path, device: str | torch.device = "cuda"):
        try:
            from models.svc.vevo2.vevo2_utils import Vevo2InferencePipeline  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Cannot import Amphion Vevo2. Put the pinned Amphion repository root on PYTHONPATH."
            ) from exc

        root = Path(vevo2_root)
        content_style_tokenizer_ckpt_path = root / "tokenizer/contentstyle_fvq16384_12.5hz"
        fmt_ckpt = root / "acoustic_modeling/fm_emilia101k_singnet7k_repa"
        vocoder = root / "vocoder"

        self.device = torch.device(device)
        self.pipeline = Vevo2InferencePipeline(
            prosody_tokenizer_ckpt_path=None,
            content_style_tokenizer_ckpt_path=str(content_style_tokenizer_ckpt_path),
            ar_cfg_path=None,
            ar_ckpt_path=None,
            fmt_cfg_path=str(fmt_ckpt / "config.json"),
            fmt_ckpt_path=str(fmt_ckpt),
            vocoder_cfg_path=str(vocoder / "config.json"),
            vocoder_ckpt_path=str(vocoder),
            device=self.device,
        )

    @torch.no_grad()
    def decode_codes_with_source_timbre(
        self,
        content_style_ids: Sequence[int],
        source_wav: str | Path,
        output_wav: str | Path,
        flow_steps: int = 32,
    ) -> str:
        from models.svc.vevo2.vevo2_utils import save_audio  # type: ignore

        codes = torch.tensor(list(map(int, content_style_ids)), dtype=torch.long, device=self.device).unsqueeze(0)
        mel = self.pipeline.code2mel(
            codes,
            timbre_ref_wav_path=str(source_wav),
            flow_matching_steps=flow_steps,
            logging=False,
        )
        wav = self.pipeline.mel2audio(mel, logging=False)
        output_wav = Path(output_wav)
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        save_audio(wav, sr=24000, output_path=str(output_wav))
        return str(output_wav)
