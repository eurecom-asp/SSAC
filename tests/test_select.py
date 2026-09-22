from ssac.select import feasible, select_top1


def row(i, p, d, hit=1, wer=0.05, sim=0.7, ratio=1.0):
    return {
        "base_pair_id": "p1",
        "candidate_id": f"c{i}",
        "target_prob": p,
        "target_prob_delta": d,
        "target_hit": hit,
        "wer": wer,
        "speaker_sim": sim,
        "duration_ratio": ratio,
        "content_style_ids": [1, 2, 3],
        "source_text": "x",
        "target_accent": "Korean",
    }


def test_gate():
    assert feasible(row(0, 0.5, 0.1))
    assert not feasible(row(0, 0.5, 0.1, wer=0.081))
    assert not feasible(row(0, 0.5, 0.1, sim=0.59))
    assert feasible(row(0, 0.5, 0.021, hit=0))
    assert not feasible(row(0, 0.5, 0.019, hit=0))


def test_lexicographic_top1():
    selected = select_top1([row(0, 0.50, 0.40), row(1, 0.51, 0.01), row(2, 0.49, 0.45)])
    assert selected[0]["candidate_id"] == "c1"
    assert selected[0]["selection_rule"] == "hard_top1"
    assert selected[0]["teacher_cs_ids"] == [1, 2, 3]
