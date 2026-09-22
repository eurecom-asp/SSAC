from pathlib import Path

import torch
import torch.nn as nn

from ssac.student import AdvancedAccentStudent, StudentConfig


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(64, 8)
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.k_proj = nn.Linear(8, 8, bias=False)
        self.v_proj = nn.Linear(8, 8, bias=False)
        self.o_proj = nn.Linear(8, 8, bias=False)

    def get_input_embeddings(self):
        return self.embed


def make_student():
    return AdvancedAccentStudent(TinyBackbone(), tokenizer=None, config=StudentConfig())


def test_advanced_adapter_checkpoint_roundtrip(tmp_path: Path):
    a = make_student()
    with torch.no_grad():
        a.accent_prompt.weight.fill_(0.125)
        a.base_model.q_proj.lora_B.fill_(0.25)
    a.save_checkpoint(tmp_path)
    assert (tmp_path / "advanced_adapter.pt").exists()

    b = make_student()
    b.load_checkpoint(tmp_path)
    assert torch.equal(a.accent_prompt.weight, b.accent_prompt.weight)
    assert torch.equal(a.base_model.q_proj.lora_B, b.base_model.q_proj.lora_B)
