"""Figures and HTML tables for the LLM notebook. Palette: the validated reference palette (categorical slots 1-3 = blue, orange, aqua;
checked all-pairs with the dataviz validator). Contrast on aqua is below 3:1 on the light surface, so identity is never colour-alone:
every series also has its own marker and line style and is named in a legend or a direct label. One axis per chart; thin marks;
text in ink, never in the series colour; status colours are paired with an icon and a word."""
import html as _html

import numpy as np

from pcsa.metrics import boot_ci
from .diagnose import frame

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e7e6e2"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]                 # validated categorical order
ARM_STYLE = {"A": dict(color=CAT[0], marker="o", ls="-"), "B": dict(color=CAT[1], marker="s", ls="-"), "C": dict(color=CAT[2], marker="^", ls="-"),
             "filter": dict(color=CAT[3], marker="D", ls="-")}
SEED_LS = ["-", "--", ":", "-."]
STATUS = {"ok": ("#0ca30c", "✓", "OK"), "info": ("#52514e", "ℹ", "INFO"), "warn": ("#fab219", "⚠", "WARN"), "fail": ("#d03b3b", "✗", "FAIL")}
MS = 6.5                                                            # marker size (pt) ~ 8-9 px


def _plt():
    import matplotlib
    import matplotlib.pyplot as plt
    return matplotlib, plt


def _style(ax, title=None, sub=None, xlabel=None, ylabel=None):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=8, length=3)
    if title:
        ax.set_title(title, loc="left", fontsize=9.5, color=INK, fontweight="bold", pad=14 if sub else 6)
    if sub:
        ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=7.5, color=INK2, va="bottom")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8, color=INK2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, color=INK2)
    return ax


def _fig(nrows, ncols, size=(4.2, 3.2), **kw):
    _, plt = _plt()
    fig, axes = plt.subplots(nrows, ncols, figsize=(size[0] * ncols, size[1] * nrows), facecolor=SURFACE, squeeze=False, **kw)
    return fig, axes


def _legend(ax, **kw):
    """Always below the axes, never over the data (the loc= hint from callers is ignored)."""
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(handles, labels, frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=min(max(len(labels), 1), 3))
    for t in leg.get_texts():
        t.set_color(INK2)
    return leg


def _roll(y, w, robust=False):
    """Rolling mean, or rolling median for heavy-tailed series (KL, critic loss), where one spike would drag a mean into a false plateau."""
    y = np.asarray(y, float)
    out = np.full(len(y), np.nan)
    for i in range(len(y)):
        seg = y[max(0, i - w + 1):i + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg):
            out[i] = np.median(seg) if robust else seg.mean()
    return out


def _ci(vals):
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    if len(v) == 1:
        return float(v[0]), np.nan, np.nan
    return boot_ci(v)


# ------------------------------------------------------------------------------------------------------------------ training
PANELS = [("reward", "Reward-model score", "higher = the reward model likes the replies more", None, False),
          ("cite_rate", "Replies that cite evidence", "share of turns with at least one citation", (0, 1), False),
          ("resp_tokens", "Reply length (tokens)", "a rising length with rising reward can mean length hacking", None, False),
          ("kl", "KL to the reference model", "log scale; dotted line = guard limit (5)", None, True),
          ("vloss", "Critic loss", "log scale; should fall as the critic fits", None, True),
          ("D", "Distance from evidence D", "how far the user ends from what the evidence supports", None, False),
          ("CW", "Confidently wrong", "share of users sure of a wrong answer", (0, 1), False),
          ("dropped", "Dialogues dropped", "cut short by an invalid user reply (survivor bias if high)", (0, 1), False)]


def plot_training(histories, title="Training", window=7):
    """histories: {label: [history per seed]} or {label: history}. Faint dots = iterations, line = rolling mean, one line style per seed."""
    fig, axes = _fig(2, 4, (4.0, 3.0))
    for ax, (key, ttl, sub, ylim, log) in zip(axes.ravel(), PANELS):
        _style(ax, ttl, sub, "iteration")
        for label, hs in histories.items():
            hs = hs if (hs and isinstance(hs[0], list)) else [hs]
            st = ARM_STYLE.get(label, ARM_STYLE["A"])
            for si, h in enumerate(hs):
                F = frame(h)
                y, x = F[key], F["iter"]
                if log:
                    y = np.where(y > 0, y, np.nan)
                ax.plot(x, y, ls="none", marker=st["marker"], ms=2.6, color=st["color"], alpha=0.18)
                ax.plot(x, _roll(y, window, robust=log), ls=SEED_LS[si % 4], lw=1.6, color=st["color"])
        if log:
            ax.set_yscale("log")
        if ylim:
            ax.set_ylim(*ylim)
        if key == "kl":
            ax.axhline(5, color=INK2, ls=":", lw=1)
        if key in ("cite_rate", "CW", "dropped"):
            ax.yaxis.set_major_formatter(lambda v, p: f"{v:.0%}")
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=ARM_STYLE.get(a, ARM_STYLE["A"])["color"], marker=ARM_STYLE.get(a, ARM_STYLE["A"])["marker"], ms=5, lw=1.6, label=f"arm {a}") for a in histories]
    n_seeds = max((len(h) if (h and isinstance(h[0], list)) else 1) for h in histories.values())
    handles += [Line2D([0], [0], color=INK2, ls=SEED_LS[i % 4], lw=1.4, label=f"seed {i}") for i in range(n_seeds)]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=8, labelcolor=INK2)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    return fig


# ------------------------------------------------------------------------------------------------------------------ gates / L1 / L2
def plot_l0(g):
    fig, axes = _fig(1, 3, (4.4, 3.4))
    ax = _style(axes[0, 0], "User vs exact Bayes", "each dot = one update; on the diagonal = Bayesian", "exact Bayesian posterior", "user's reported probability")
    pts = g.get("points")
    if pts:
        ax.plot(pts["bayes"], pts["user"], ls="none", marker="o", ms=3.2, alpha=0.35, color=CAT[0], label=f"updates (Spearman {g['spearman']:.2f})")
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--", label="perfect Bayesian")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); _legend(ax)
    ax = _style(axes[0, 1], "Spec gate", "bar = result, line = required", None, "value")
    vals, need, names = [g["spearman"], g["direction"]], [0.8, 0.9], ["Spearman", "direction"]
    ax.bar(names, np.nan_to_num(vals), width=0.45, color=[CAT[0] if v >= n else STATUS["fail"][0] for v, n in zip(np.nan_to_num(vals), need)])
    for i, n in enumerate(need):
        ax.hlines(n, i - 0.32, i + 0.32, color=INK, lw=1.6)
        ax.text(i, min(np.nan_to_num(vals)[i], 1) + 0.02, f"{vals[i]:.2f}", ha="center", fontsize=8, color=INK)
    ax.set_ylim(0, 1.1)
    a = g.get("assertion", {})
    ax = _style(axes[0, 2], "Assertion probe", "user shown NO evidence, only a confident wrong claim", "shift toward the asserted answer", "users")
    if a.get("shifts"):
        ax.hist(a["shifts"], bins=np.linspace(-0.3, 0.6, 19), color=CAT[1], rwidth=0.88)
    ax.axvline(0.05, color=INK, lw=1.6, ls="--", label="limit 0.05")
    ax.axvline(a.get("mean_shift", 0), color=INK2, lw=1.2, label=f"mean {a.get('mean_shift', float('nan')):+.3f}")
    _legend(ax, loc="upper right")
    fig.tight_layout()
    return fig


def plot_l0_lab(lab):
    """Prompt-style / view variants side by side. Bars = result, line = what is required, glyph = pass or fail."""
    names = list(lab)
    fig, axes = _fig(1, 4, (4.0, 3.4))
    specs = [("Spearman vs exact Bayes", lambda g: g["spearman"], 0.8, "higher is better", True),
             ("Moves the right way", lambda g: g["direction"], 0.9, "share of shown items", True),
             ("Valid JSON replies", lambda g: 1 - g["dropped_dialogues"] / max(g.get("n_dialogues", 1), 1), 0.9, "share not dropped", True),
             ("Assertion shift", lambda g: g["assertion"]["mean_shift"], 0.05, "toward an uncited claim; lower is better", False)]
    for ax, (ttl, fn, need, sub, higher) in zip(axes[0], specs):
        _style(ax, ttl, sub, None, None)
        vals = np.array([np.nan_to_num(fn(lab[n])) for n in names])
        ax.bar(range(len(names)), vals, width=0.55, color=CAT[0])
        ax.axhline(need, color=INK, lw=1.6, ls="--")
        for i, v in enumerate(vals):
            ok = v >= need if higher else v <= need
            ax.text(i, max(v, 0) + 0.02 * (1 if higher else max(need, 0.05) * 4), f"{'\u2713' if ok else '\u2717'} {v:.2f}", ha="center", fontsize=8, color=STATUS['ok' if ok else 'fail'][0], fontweight="bold")
        ax.set_xticks(range(len(names))); ax.set_xticklabels([n.replace("/", "\n") for n in names], fontsize=7)
        if higher:
            ax.set_ylim(0, 1.15)
    fig.suptitle("L0 lab: which user setup passes?  (dashed line = requirement)", x=0.01, ha="left", fontsize=11, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def plot_l1(l1):
    rms = list(l1["per_rm"])
    ks = sorted(next(iter(l1["per_rm"].values()))["k"])
    fig, axes = _fig(1, 2, (6.0, 3.6))
    ax = _style(axes[0, 0], "Does the reward model pay for selective evidence?", "selective-k minus balanced-k summed reward; above 0 = pays for steering",
                "evidence turns before agreeing (k)", "reward difference (95% CI)")
    w = 0.8 / len(rms)
    for i, name in enumerate(rms):
        st = ARM_STYLE[["A", "B", "C", "filter"][i % 4]]
        m = [l1["per_rm"][name]["k"][k] for k in ks]
        x = np.arange(len(ks)) + (i - (len(rms) - 1) / 2) * w
        ax.errorbar(x, [v[0] for v in m], yerr=[[v[0] - v[1] for v in m], [v[2] - v[0] for v in m]], fmt=st["marker"], ms=MS, color=st["color"], capsize=3, lw=1.4, label=name)
    ax.axhline(0, color=INK, lw=1)
    ax.set_xticks(range(len(ks))); ax.set_xticklabels(ks); _legend(ax, loc="best")
    ax = _style(axes[0, 1], "Agreeing right after a correction", "agree-after-correction minus agree-at-turn-1; above 0 = corrections raise the payoff of agreeing", None, "reward difference (95% CI)")
    for i, name in enumerate(rms):
        st = ARM_STYLE[["A", "B", "C", "filter"][i % 4]]
        c = l1["per_rm"][name]["agree_after_correction_minus_turn1"]
        ax.errorbar([i], [c[0]], yerr=[[c[0] - c[1]], [c[2] - c[0]]], fmt=st["marker"], ms=MS + 1, color=st["color"], capsize=3, lw=1.4, label=name)
    ax.axhline(0, color=INK, lw=1); ax.set_xticks(range(len(rms))); ax.set_xticklabels(rms); ax.set_xlim(-0.6, len(rms) - 0.4); _legend(ax, loc="best")
    fig.tight_layout()
    return fig


def plot_l2(rows):
    keys = [("agree", "Agree share"), ("selective", "Selective share"), ("SB", "Selection bias SB"), ("D", "Distance from evidence D"), ("CW", "Confidently wrong"), ("DS", "Disclosure shift DS")]
    stages = list(rows)
    fig, axes = _fig(2, 3, (3.8, 3.0))
    for ax, (k, t) in zip(axes.ravel(), keys):
        _style(ax, t, None, None, None)
        for i, s in enumerate(stages):
            v = np.array([r[k] for r in rows[s]], float)
            m, lo, hi = _ci(v)
            ax.bar(i, np.nanmean(v), width=0.5, color=CAT[0], alpha=0.9)
            ax.plot([i] * len(v), v, ls="none", marker="o", ms=4, color=INK2, alpha=0.8)
            if np.isfinite(lo):
                ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], color=INK, capsize=3, lw=1.2)
        ax.set_xticks(range(len(stages))); ax.set_xticklabels(stages, fontsize=8)
    fig.suptitle("L2: the same questions and user, four training stages (bars = mean, dots = seeds)", x=0.01, ha="left", fontsize=11, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


# ------------------------------------------------------------------------------------------------------------------ L3 and after
EVAL_KEYS = [("agree", "Agree share"), ("selective", "Selective share"), ("SB", "Selection bias SB"), ("D", "Distance from evidence D"), ("CW", "Confidently wrong"), ("DS", "Disclosure shift DS")]


def plot_arms(arms, title="Held-out evaluation by arm (full horizon, same questions)"):
    fig, axes = _fig(2, 3, (3.8, 3.0))
    names = list(arms)
    for ax, (k, t) in zip(axes.ravel(), EVAL_KEYS):
        _style(ax, t, None, None, None)
        for i, a in enumerate(names):
            v = np.array([r["eval"][0][k] for r in arms[a]], float)
            st = ARM_STYLE.get(a, ARM_STYLE["A"])
            ax.bar(i, np.nanmean(v) if np.isfinite(v).any() else 0, width=0.5, color=st["color"], alpha=0.9, label=f"{a} (n={len(v)})")
            ax.plot([i] * len(v), v, ls="none", marker=st["marker"], ms=4.5, color=INK, alpha=0.85)
            m, lo, hi = _ci(v)
            if np.isfinite(lo):
                ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], color=INK, capsize=3, lw=1.2)
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, fontsize=8, labelcolor=INK2)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    return fig


def plot_critic(arms):
    fig, axes = _fig(1, 1, (6.2, 3.6))
    ax = _style(axes[0, 0], "What the critic thinks of each user state", "arm C should value settled users more and correction states less (the spec's prediction)", None, "critic value")
    groups = [("value_settled", "settled user"), ("value_unsettled", "unsettled user"), ("value_after_correction", "after a correction"), ("value_no_correction", "no correction")]
    w = 0.8 / len(arms)
    for i, a in enumerate(arms):
        st = ARM_STYLE.get(a, ARM_STYLE["A"])
        vals = [np.nanmean([r["critic"].get(k, np.nan) for r in arms[a]]) for k, _ in groups]
        ax.bar(np.arange(len(groups)) + (i - (len(arms) - 1) / 2) * w, np.nan_to_num(vals), width=w * 0.9, color=st["color"], label=f"arm {a}")
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[1] for g in groups]); ax.axhline(0, color=INK, lw=1); _legend(ax, loc="best")
    fig.tight_layout()
    return fig


def plot_l4(pref, probe, steer):
    fig, axes = _fig(1, 4, (4.0, 3.3))
    ax = _style(axes[0, 0], "What each policy prefers", "within-state Spearman between a reply's probability and a readout", None, "rank correlation")
    reads = [("pred_restate", "predicts user restates"), ("neg_pred_entropy", "predicts settled user"), ("critic_value", "critic value")]
    arms = list(pref)
    w = 0.8 / len(arms)
    for i, a in enumerate(arms):
        st = ARM_STYLE.get(a, ARM_STYLE["A"])
        ax.bar(np.arange(len(reads)) + (i - (len(arms) - 1) / 2) * w, [np.nan_to_num(pref[a].get(k, np.nan)) for k, _ in reads], width=w * 0.9, color=st["color"], label=f"arm {a}")
    ax.set_xticks(range(len(reads))); ax.set_xticklabels([r[1] for r in reads], fontsize=7, rotation=12); ax.axhline(0, color=INK, lw=1); _legend(ax, loc="best")
    ax = _style(axes[0, 1], "Probes on the policy's activations", "held-out questions; baseline = always guess the most common answer", None, None)
    ax.bar([0], [probe["conf_r"]], width=0.5, color=CAT[0]); ax.text(0, probe["conf_r"] + 0.02, f"r = {probe['conf_r']:.2f}", ha="center", fontsize=8, color=INK)
    ax.bar([1.2], [probe["top_acc"]], width=0.5, color=CAT[0]); ax.hlines(probe["top_base"], 0.9, 1.5, color=INK, lw=1.6)
    ax.text(1.2, probe["top_acc"] + 0.02, f"{probe['top_acc']:.0%}", ha="center", fontsize=8, color=INK)
    ax.set_xticks([0, 1.2]); ax.set_xticklabels(["user confidence (r)", "user top answer (acc.)"], fontsize=7.5); ax.set_ylim(0, 1.1)
    al = sorted(steer)
    for j, (key, ttl) in enumerate((("SB", "Steering: selection bias SB"), ("CW", "Steering: confidently wrong"))):
        ax = _style(axes[0, 2 + j], ttl, "add alpha x (confidence direction) while generating; negative = policy thinks user is less sure", "alpha", key)
        ax.plot(al, [steer[a][key] for a in al], marker="o", ms=MS, lw=1.6, color=CAT[2 if j else 0])
    fig.tight_layout()
    return fig


def plot_l5(filt, arms):
    fig, axes = _fig(1, 3, (4.2, 3.3))
    names = ["B", "C", "filter"]
    data = {"B": arms["B"], "C": arms["C"], "filter": filt}
    for ax, (k, t) in zip(axes[0, :2], (("D", "Distance from evidence D"), ("CW", "Confidently wrong"))):
        _style(ax, t, "filtered arm should fall to arm B or below", None, None)
        for i, n in enumerate(names):
            v = np.array([r["eval"][0][k] for r in data[n]], float)
            st = ARM_STYLE[n]
            ax.bar(i, np.nanmean(v), width=0.5, color=st["color"], label={"B": "B (gamma 0)", "C": "C (gamma 0.95)", "filter": "C + filter"}[n])
            ax.plot([i] * len(v), v, ls="none", marker=st["marker"], ms=4.5, color=INK)
        ax.set_xticks(range(3)); ax.set_xticklabels(["B", "C", "C + filter"])
    _legend(axes[0, 0], loc="best")
    ax = _style(axes[0, 2], "What the filter did while training", "per iteration; fallback = no draw was acceptable", "iteration", "share of states")
    for j, r in enumerate(filt):
        fl = [h.get("filter") for h in r["history"] if h.get("filter")]
        if len(fl) > 1:
            d = lambda key: np.diff([0] + [f[key] for f in fl])
            ax.plot(range(len(fl)), d("accepted") / np.maximum(d("drawn"), 1), color=CAT[2], ls=SEED_LS[j % 4], lw=1.6, marker="o", ms=3, label=f"accepted draws (seed {j})")
            ax.plot(range(len(fl)), d("fallbacks") / np.maximum(d("states"), 1), color=CAT[3], ls=SEED_LS[j % 4], lw=1.6, marker="D", ms=3, label=f"fallback states (seed {j})")
    ax.set_ylim(0, 1); _legend(ax, loc="best")
    fig.tight_layout()
    return fig


def plot_l6(v6):
    lam = v6["lambda"]
    fig, axes = _fig(1, 3, (4.0, 3.3))
    for ax, (key, t, sub) in zip(axes[0], (("flag_rate", "Judge flag rate", "should fall as lambda rises"), ("SB", "Selection bias SB", "should stay up if the policy hides it"), ("D", "Distance from evidence D", "should stay above baseline"))):
        _style(ax, t, sub, "lambda (cost per flag)", None)
        ax.plot(lam, v6[key], marker="o", ms=MS, lw=1.6, color=CAT[0])
    fig.tight_layout()
    return fig


def plot_timeline(rows):
    _, plt = _plt()
    done = [r for r in rows if r.get("minutes")]
    fig, axes = _fig(1, 1, (7.5, max(1.6, 0.34 * len(done) + 1.0)))
    ax = _style(axes[0, 0], "Where the time went", "minutes per finished part", "minutes", None)
    ax.barh(range(len(done)), [r["minutes"] for r in done], color=CAT[0], height=0.55)
    ax.set_yticks(range(len(done))); ax.set_yticklabels([r["part"] for r in done], fontsize=8); ax.invert_yaxis()
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------------------------------------------------------ HTML tables
def findings_html(findings):
    rows = []
    for f in findings:
        f = f if isinstance(f, dict) else f.to_dict()
        col, icon, word = STATUS[f["severity"]]
        act = f"<div style='color:{INK2}'>&rarr; {_html.escape(f['action'])}</div>" if f.get("action") else ""
        rows.append(f"<tr><td style='white-space:nowrap;padding:4px 10px 4px 0;color:{INK}'><span style='color:{col};font-weight:700;font-size:1.1em'>{icon}</span> {word}</td>"
                    f"<td style='padding:4px 0;color:{INK}'><b>{_html.escape(f.get('stage') or '')}{' · ' if f.get('stage') else ''}{_html.escape(f['title'])}</b>"
                    f"<div style='color:{INK2}'>{_html.escape(f['evidence'])}</div>{act}</td></tr>")
    return f"<table style='border-collapse:collapse;font:13px system-ui;background:{SURFACE};padding:6px'>{''.join(rows)}</table>"


def status_html(rows):
    icon = {"done": ("#0ca30c", "✓"), "in progress (checkpoint)": ("#fab219", "◐"), "not started": ("#52514e", "○")}
    body = ""
    for r in rows:
        col, ic = icon[r["state"]]
        mins = "" if r["minutes"] is None else f"{r['minutes']:.1f} min"
        body += (f"<tr><td style='padding:2px 10px;color:{INK}'><span style='color:{col};font-weight:700'>{ic}</span> {_html.escape(r['part'])}</td>"
                 f"<td style='padding:2px 10px;color:{INK2}'>{r['state']}</td><td style='padding:2px 10px;color:{INK2}'>{mins}</td><td style='padding:2px 10px;color:{INK2}'>{r['saved_at']}</td></tr>")
    return f"<table style='border-collapse:collapse;font:13px system-ui;background:{SURFACE}'><tr style='color:{INK2}'><th align=left style='padding:2px 10px'>part</th><th align=left>state</th><th align=left>time</th><th align=left>saved</th></tr>{body}</table>"


def samples_html(eps, n=8, chars=420):
    """A table of actual replies: the cheapest way to see what the policy is doing (and to catch reward hacking by eye)."""
    rows = ""
    for e in eps[:n]:
        for t in e.turns[:2]:
            rows += (f"<tr><td style='padding:3px 8px;color:{INK2};white-space:nowrap'>lean {e.q.entities[e.q.lean]}<br>truth {e.q.entities[e.q.hstar]}</td>"
                     f"<td style='padding:3px 8px;color:{INK2}'>{t['category']}<br>cited {t['cited']}</td>"
                     f"<td style='padding:3px 8px;color:{INK}'>{_html.escape(t['text'][:chars])}</td></tr>")
    return f"<table style='border-collapse:collapse;font:12px system-ui;background:{SURFACE}'><tr style='color:{INK2}'><th align=left>question</th><th align=left>turn</th><th align=left>reply</th></tr>{rows}</table>"
