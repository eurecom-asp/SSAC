from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import ACCENTS, ACCENT_TO_ID


def _require_amphion_qwen_utils():
    try:
        from models.svc.vevo2.qwen_utils import (  # type: ignore
            extract_content_style_ids,
            gen_chat_prompt,
            gen_chat_response,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Amphion is required. Checkout the experiment commit and put its repository root "
            "on PYTHONPATH before running SSAC."
        ) from exc
    return gen_chat_prompt, gen_chat_response, extract_content_style_ids


@dataclass
class AdvancedStudentExample:
    pair_id: str
    source_text: str
    source_prosody_ids: list[int]
    teacher_cs_ids: list[int]
    target_accent: str
    negative_cs_ids: list[int] | None = None
    source_speaker: str = ""


@dataclass
class StudentConfig:
    prompt_length: int = 32
    max_sequence_length: int = 1024
    lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    shared_lora_rank: int = 32
    shared_lora_alpha: float = 64.0
    accent_lora_rank: int = 0
    accent_lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    use_phone_prefix: bool = False
    phone_prefix_length: int = 24
    use_phone_cross_attention: bool = False
    phone_cross_attention_dim: int = 128
    use_deep_film: bool = False
    accent_loss_weight: float = 0.0
    content_loss_weight: float = 0.0
    preference_loss_weight: float = 0.0
    preference_beta: float = 0.1

    def validate_final_icassp(self) -> None:
        if self.prompt_length != 32:
            raise ValueError("Final ICASSP config uses prompt_length=32")
        if self.shared_lora_rank != 32:
            raise ValueError("Final ICASSP config uses shared_lora_rank=32")
        if self.accent_lora_rank != 0:
            raise ValueError("Final ICASSP config uses accent_lora_rank=0")
        if self.use_phone_prefix or self.use_phone_cross_attention or self.use_deep_film:
            raise ValueError("Phone-prefix/cross-attention/deep-FiLM are disabled in final ICASSP config")
        if any(
            x != 0.0
            for x in (
                self.accent_loss_weight,
                self.content_loss_weight,
                self.preference_loss_weight,
            )
        ):
            raise ValueError("Final ICASSP objective is token CE/NLL only")


class AccentPromptBank(nn.Module):
    """One learned continuous prompt per categorical target accent."""

    def __init__(self, n_accents: int, prompt_length: int, hidden_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_accents, prompt_length, hidden_size))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, accent_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[accent_ids]


class LoRALinear(nn.Module):
    """Minimal shared LoRA wrapper for a frozen Linear projection."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        self.rank = rank
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return self.base(x) + delta * self.scaling


def inject_shared_lora(
    module: nn.Module,
    target_modules: Sequence[str],
    rank: int,
    alpha: float,
    dropout: float,
) -> int:
    target = set(target_modules)
    replaced = 0
    for name, child in list(module.named_children()):
        if name in target and isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
            replaced += 1
        else:
            replaced += inject_shared_lora(child, target_modules, rank, alpha, dropout)
    return replaced


class AdvancedAccentStudent(nn.Module):
    """
    Vevo2 AR student used by the ICASSP system.

    Active conditioning in the final configuration is:
      text-prefix embeddings + learned 32-position target-accent prompt.
    Source audio is not an AR input; it is used later as the frozen acoustic/timbre reference.
    """

    def __init__(
        self,
        base_model: nn.Module,
        tokenizer,
        config: StudentConfig | None = None,
    ):
        super().__init__()
        self.config = config or StudentConfig()
        self.config.validate_final_icassp()
        self.base_model = base_model
        self.tokenizer = tokenizer

        for p in self.base_model.parameters():
            p.requires_grad = False

        hidden_size = int(self.base_model.get_input_embeddings().embedding_dim)
        self.accent_prompt = AccentPromptBank(len(ACCENTS), self.config.prompt_length, hidden_size)
        n = inject_shared_lora(
            self.base_model,
            self.config.lora_target_modules,
            rank=self.config.shared_lora_rank,
            alpha=self.config.shared_lora_alpha,
            dropout=self.config.lora_dropout,
        )
        if n == 0:
            raise RuntimeError(
                f"No LoRA targets found among {self.config.lora_target_modules}; "
                "check the Vevo2/Qwen backbone version."
            )

    @classmethod
    def from_pretrained(
        cls,
        ar_checkpoint: str,
        config: StudentConfig | None = None,
        torch_dtype: torch.dtype | str = torch.bfloat16,
        device: str | torch.device = "cuda",
    ) -> "AdvancedAccentStudent":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(
            ar_checkpoint,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(ar_checkpoint, local_files_only=True)
        student = cls(model, tokenizer, config=config)
        return student.to(device)

    def _prefix_ids(self, source_text: str) -> list[int]:
        gen_chat_prompt, _, _ = _require_amphion_qwen_utils()
        prompt = gen_chat_prompt(
            source_text,
            add_assistant_token=True,
            follow_prosody_instruction=False,
        )
        return self.tokenizer.encode(prompt, add_special_tokens=True)

    def _target_ids(self, cs_ids: Sequence[int]) -> list[int]:
        _, gen_chat_response, _ = _require_amphion_qwen_utils()
        response = gen_chat_response(None, list(map(int, cs_ids)), is_full_response=True)
        return self.tokenizer.encode(response, add_special_tokens=False)

    def _single_sequence(self, ex: AdvancedStudentExample) -> tuple[torch.Tensor, torch.Tensor]:
        if ex.target_accent not in ACCENT_TO_ID:
            raise ValueError(f"Unknown target accent: {ex.target_accent}")
        if ex.source_prosody_ids:
            raise ValueError("Final ICASSP system disables source-prosody conditioning")

        device = next(self.parameters()).device
        embed = self.base_model.get_input_embeddings()
        prefix_ids = torch.tensor(self._prefix_ids(ex.source_text), dtype=torch.long, device=device)
        target_ids = torch.tensor(self._target_ids(ex.teacher_cs_ids), dtype=torch.long, device=device)
        accent_id = torch.tensor([ACCENT_TO_ID[ex.target_accent]], dtype=torch.long, device=device)

        prefix_emb = embed(prefix_ids)
        accent_emb = self.accent_prompt(accent_id)[0]
        target_emb = embed(target_ids)
        inputs = torch.cat([prefix_emb, accent_emb, target_emb], dim=0)

        labels = torch.full((inputs.shape[0],), -100, dtype=torch.long, device=device)
        target_start = prefix_emb.shape[0] + accent_emb.shape[0]
        labels[target_start:] = target_ids

        if inputs.shape[0] > self.config.max_sequence_length:
            raise ValueError(
                f"Sequence {ex.pair_id} has length {inputs.shape[0]} > "
                f"max_sequence_length={self.config.max_sequence_length}"
            )
        return inputs, labels

    def _batch_build(self, examples: Sequence[AdvancedStudentExample]):
        seqs = [self._single_sequence(ex) for ex in examples]
        max_len = max(x.shape[0] for x, _ in seqs)
        hidden = seqs[0][0].shape[-1]
        device = seqs[0][0].device
        dtype = seqs[0][0].dtype

        inputs = torch.zeros(len(seqs), max_len, hidden, device=device, dtype=dtype)
        attention = torch.zeros(len(seqs), max_len, device=device, dtype=torch.long)
        labels = torch.full((len(seqs), max_len), -100, device=device, dtype=torch.long)
        lengths = []
        for i, (emb, lab) in enumerate(seqs):
            n = emb.shape[0]
            inputs[i, :n] = emb
            attention[i, :n] = 1
            labels[i, :n] = lab
            lengths.append(n)
        return inputs, attention, labels, lengths

    def forward(self, examples: Sequence[AdvancedStudentExample]):
        inputs, attention, labels, _ = self._batch_build(examples)
        return self.base_model(inputs_embeds=inputs, attention_mask=attention, labels=labels)

    @torch.no_grad()
    def per_example_nll(self, examples: Sequence[AdvancedStudentExample]) -> torch.Tensor:
        inputs, attention, labels, _ = self._batch_build(examples)
        logits = self.base_model(inputs_embeds=inputs, attention_mask=attention).logits
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        token_loss = F.cross_entropy(
            shift_logits.transpose(1, 2),
            shift_labels,
            ignore_index=-100,
            reduction="none",
        )
        mask = shift_labels.ne(-100)
        return (token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

    @torch.no_grad()
    def generate_advanced_codes(
        self,
        source_text: str,
        target_accent: str,
        source_prosody_ids: Sequence[int] | None = None,
        *,
        max_new_tokens: int = 500,
        min_new_tokens: int = 15,
        top_k: int = 25,
        top_p: float = 0.8,
        temperature: float = 1.0,
        do_sample: bool = True,
    ) -> list[int]:
        if source_prosody_ids:
            raise ValueError("Final ICASSP generation uses --no-use-source-prosody")
        if target_accent not in ACCENT_TO_ID:
            raise ValueError(f"Unknown target accent: {target_accent}")

        device = next(self.parameters()).device
        embed = self.base_model.get_input_embeddings()
        prefix_ids = torch.tensor(self._prefix_ids(source_text), dtype=torch.long, device=device)
        accent_id = torch.tensor([ACCENT_TO_ID[target_accent]], dtype=torch.long, device=device)
        inputs = torch.cat([embed(prefix_ids), self.accent_prompt(accent_id)[0]], dim=0).unsqueeze(0)
        attention = torch.ones(inputs.shape[:2], dtype=torch.long, device=device)

        generated = self.base_model.generate(
            inputs_embeds=inputs,
            attention_mask=attention,
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
        text = self.tokenizer.decode(generated[0], skip_special_tokens=False)
        _, _, extract_content_style_ids = _require_amphion_qwen_utils()
        ids = extract_content_style_ids(text).tolist()
        if not ids:
            raise RuntimeError("AR generation returned no <|content_style_*|> tokens")
        return [int(x) for x in ids]

    def trainable_parameter_names(self) -> set[str]:
        return {name for name, p in self.named_parameters() if p.requires_grad}

    def save_checkpoint(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        names = self.trainable_parameter_names()
        state = {k: v.detach().cpu() for k, v in self.state_dict().items() if k in names}
        # `advanced_adapter.pt` is the filename used by the original experiment tree.
        torch.save(state, output_dir / "advanced_adapter.pt")
        (output_dir / "student_config.json").write_text(
            json.dumps(asdict(self.config), indent=2), encoding="utf-8"
        )

    def load_checkpoint(self, checkpoint_dir: str | Path) -> tuple[list[str], list[str]]:
        checkpoint_dir = Path(checkpoint_dir)
        path = checkpoint_dir
        if path.is_dir():
            legacy = path / "advanced_adapter.pt"
            release = path / "trainable_state.pt"
            if legacy.exists():
                path = legacy
            elif release.exists():  # compatibility with early release drafts
                path = release
            else:
                raise FileNotFoundError(
                    f"No advanced_adapter.pt found under {checkpoint_dir}"
                )
        raw = torch.load(path, map_location="cpu")
        state = raw
        if isinstance(raw, dict):
            for key in ("state_dict", "model_state_dict", "adapter_state_dict"):
                if key in raw and isinstance(raw[key], dict):
                    state = raw[key]
                    break
        if not isinstance(state, dict):
            raise TypeError(f"Unsupported checkpoint object in {path}: {type(state)!r}")
        # Common wrappers used by DDP/Accelerate do not change parameter semantics.
        if state and all(isinstance(k, str) and k.startswith("module.") for k in state):
            state = {k[len("module."):]: v for k, v in state.items()}
        incompatible = self.load_state_dict(state, strict=False)
        return list(incompatible.missing_keys), list(incompatible.unexpected_keys)
