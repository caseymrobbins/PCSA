"""Every figure builds, from synthetic data shaped like the real results, and passes the validated palette through unchanged."""
import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from pcsa_llm import plots as PL
from pcsa_llm.htmlreport import build_report
from pcsa_llm.store import Store


def hist(n=30, seed=0):
    rng = np.random.default_rng(seed)
    return [{"iter": i, "minutes": i * 1.2, "reward": -8 + 0.2 * i + rng.normal(), "summary": {"D": 1.3 + rng.normal(0, .2), "CW": float(rng.random() * .4),
             "SB": float("nan") if i > 15 else .4, "agree": .5, "selective": .2, "flag_rate": .3 - .005 * i}, "ppo": {"vloss": 50 * np.exp(-i / 10), "kl": .1},
             "diag": {"cite_rate": max(.6 - .03 * i, 0), "resp_tokens": 120 + i, "kl": 0.01 * i + (30 if i == 12 else 0), "dropped_frac": .05, "dead_frac": 0.0},
             "filter": {"states": 8 * (i + 1), "drawn": 32 * (i + 1), "accepted": 20 * (i + 1), "fallbacks": 2 * (i + 1)}} for i in range(n)]


def arm(a, n_seeds=2):
    return [{"arm": a, "seed": s, "history": hist(seed=s), "eval": [{"agree": .5, "selective": .2, "balanced": .1, "SB": .3, "D": 1.3, "CW": .2, "DS": 1.1, "H": .8}],
             "critic": {"value_settled": 1.0, "value_unsettled": .2, "value_after_correction": -.3, "value_no_correction": .4}} for s in range(n_seeds)]


def test_the_palette_matches_the_validated_reference():
    assert PL.CAT[:3] == ["#2a78d6", "#eb6834", "#1baf7a"] and PL.SURFACE == "#fcfcfb"
    assert {k: v[0] for k, v in PL.STATUS.items() if k != "info"} == {"ok": "#0ca30c", "warn": "#fab219", "fail": "#d03b3b"}
    assert len({(v["marker"], v["color"]) for v in PL.ARM_STYLE.values()}) == len(PL.ARM_STYLE)       # no two arms share colour or marker


def test_every_figure_builds_and_saves(tmp_path):
    arms = {a: arm(a) for a in "ABC"}
    gate = {"pass": True, "spearman": .93, "direction": .97, "dropped_dialogues": 0, "points": {"bayes": np.random.rand(50).tolist(), "user": np.random.rand(50).tolist()},
            "assertion": {"pass": True, "mean_shift": .01, "shifts": np.random.normal(0, .03, 30).tolist(), "share_shifted": .1, "dropped": 0, "n": 30}}
    l1 = {"per_rm": {r: {"k": {k: (.5, .1, .9) for k in (1, 2, 3, 5)}, "agree_after_correction_minus_turn1": (.2, -.1, .5)} for r in ("skywork", "tulu")}}
    figs = [PL.plot_training({a: [r["history"] for r in rs] for a, rs in arms.items()}), PL.plot_l0(gate), PL.plot_l1(l1),
            PL.plot_l2({s: [a["eval"][0] for a in arm("A")] for s in ("base", "sft", "dpo", "final")}), PL.plot_arms(arms), PL.plot_critic(arms),
            PL.plot_l4({a: {"pred_restate": .1, "neg_pred_entropy": .05, "critic_value": .0} for a in "ABC"}, {"conf_r": .5, "top_acc": .6, "top_base": .4},
                       {-2.0: {"SB": .2, "CW": .1}, 0: {"SB": .3, "CW": .2}, 2.0: {"SB": .4, "CW": .3}}),
            PL.plot_l5(arm("filter"), arms), PL.plot_l6({"lambda": [.5, 2.0], "flag_rate": [.3, .1], "SB": [.4, .4], "D": [1.5, 1.4]}),
            PL.plot_timeline([{"part": "L1", "minutes": 4.0}, {"part": "L3/A/s0", "minutes": 50.0}])]
    s = Store(tmp_path)
    for i, f in enumerate(figs):
        assert f.get_facecolor()[:3] == matplotlib.colors.to_rgb(PL.SURFACE)
        assert s.fig(f, f"fig{i}").exists()


def test_html_tables_and_the_self_contained_report(tmp_path):
    from pcsa_llm.diagnose import Finding
    s = Store(tmp_path)
    s.save("L1", {"x": 1}, minutes=2.0)
    s.add_findings("L3", [Finding("fail", "Policy stopped citing evidence", "55% -> 2%", "read the samples", "A"), Finding("ok", "Fine", "x")])
    import matplotlib.pyplot as plt
    f, ax = plt.subplots(); ax.plot([1, 2]); s.fig(f, "demo")
    out = build_report(s, planned=["L1", "L3/A/s0"])
    txt = out.read_text()
    assert "Needs attention" in txt and "Policy stopped citing evidence" in txt and "data:image/png;base64," in txt and "not started" in txt
    html = PL.findings_html([Finding("warn", "t", "e", "a")])
    assert "WARN" in html and "⚠" in html and "&rarr; a" in html            # status = icon + word + colour, never colour alone


def test_assertion_histogram_shows_every_user_even_when_shifts_are_large(tmp_path):
    """Regression: bins ended at +0.6, so users who moved by more than that (the majority in the first real run) were not drawn."""
    shifts = [0.0] * 4 + [0.64, 0.7, 0.72, 0.75, 0.78, 0.8, 0.8, 0.79]
    gate = {"pass": False, "spearman": .75, "direction": .92, "dropped_dialogues": 0, "points": None,
            "assertion": {"pass": False, "mean_shift": float(np.mean(shifts)), "shifts": shifts, "share_shifted": .8, "dropped": 0, "n": len(shifts)}}
    fig = PL.plot_l0(gate)
    ax = fig.axes[2]
    drawn = sum(p.get_height() for p in ax.patches)
    assert drawn == len(shifts)                                              # every user is in a bar
    assert ax.get_xlim()[1] >= max(shifts)                                   # and inside the visible axis
