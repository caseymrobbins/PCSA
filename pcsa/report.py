"""Tables and figures for the notebook. Every number is a seed-level mean with a 95% bootstrap CI."""
import numpy as np
import pandas as pd

from .metrics import boot_ci

BADGE = {"supports": "SUPPORTS", "weakens": "WEAKENS", "neutral": "neutral", "invalid": "INVALID"}


def ci(x, nd=2):
    m, lo, hi = boot_ci(np.asarray(x, float)[np.isfinite(x)])
    return f"{m:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


GATED = ("S2", "S3", "S4", "S6")        # results that interpret a learner; blocked while Gate 0 fails


def verdict_table(named, gate0_passed=True):
    """named: {test: verdict-dict-or-{key: verdict-dict}}. Learner-based tests read UNINTERPRETABLE while Gate 0 fails."""
    rows = []
    for test, v in named.items():
        items = [(None, v)] if "status" in v else list(v.items())
        for key, d in items:
            res = BADGE[d["status"]]
            if d["claim"] == "C1" and d["status"] == "neutral" and d.get("beats_balanced") and not d.get("beyond_settling"):
                res = "narrowed: pays for settling the user, no premium over settle-toward-truth"
            if test in GATED and not gate0_passed:
                res = f"UNINTERPRETABLE (Gate 0 failed; raw: {res})"
            rows.append({"test": test, "condition": "" if key is None else str(key), "claim": d["claim"], "result": res})
    return pd.DataFrame(rows)


def gate0_table(gates, myopic):
    df = pd.DataFrame(gates)[["reward", "action_match", "sign_match", "learner_return", "best_script_return", "optimum", "return_ratio", "pass"]]
    df["myopic_equivalence_in_solver"] = myopic["pass"]
    return df.round(3)


def s1_table(r, user, reward):
    d = r["res"][user][reward]
    cols = ["ret", "D", "CW", "H", "SB", "DS"] + (["kappa_final"] if user == "deference" else [])
    return pd.DataFrame({p: {c: ci(d[p][c]) for c in cols} for p in d}).T


def s1_pairs(r):
    rows = []
    for (u, rw), v in r["verdict"].items():
        for k, x in v["per_k"].items():
            rows.append({"user": u, "reward": rw, "k": k, "selective - balanced": f"{x['sel_minus_bal']:.2f} [{x['ci'][0]:.2f}, {x['ci'][1]:.2f}]",
                         "selective - infogain": f"{x['sel_minus_ig']:.2f} [{x['ci_ig'][0]:.2f}, {x['ci_ig'][1]:.2f}]"})
    return pd.DataFrame(rows)


def s2_table(r, user, reward):
    d = r["res"][user][reward]
    cols = ["ret", "D", "CW", "share_agree", "share_bal", "share_sel", "sel_minus_bal", "sel_share_ev", "A_imm", "A_fut", "corr_fut_entropy", "corr_fut_restate"]
    return pd.DataFrame({g: {c: ci(d[g][c]) for c in cols} for g in d}).T.rename_axis("gamma")


def s2_agency(r, reward="general"):
    d = r["res"].get("deference", {}).get(reward)
    if d is None:
        return pd.DataFrame()
    cols = ["kappa_final", "req_rate", "excess_drop", "rho_balanced", "rho_agree", "C_balanced", "Cp_balanced", "Cp_agree"]
    return pd.DataFrame({g: {c: ci(d[g][c]) for c in cols} for g in d}).T.rename_axis("gamma")


def s3_table(r):
    rows = []
    for u, ur in r["res"].items():
        for rw, rr in ur.items():
            for g, gg in rr.items():
                for t0, d in gg.items():
                    rows.append({"user": u, "reward": rw, "gamma": g, "t0": t0, **{k: ci(d[k], 3) for k in ("dV", "dA", "rise_did", "d_kappa", "d_excess")}})
    return pd.DataFrame(rows)


def s4_table(r, user, reward):
    d = r["res"][user][reward]
    base = r["baseline"][(user, reward)]
    cols = ["ret", "D", "CW", "fa", "fr", "empty"] + (["rho", "Cp"] if user == "deference" else [])
    out = {"always-balanced": {"ret": ci(base["ret"]), "D": ci(base["D"]), "CW": ci(base["CW"])}}
    for st, v in d.items():
        out[f"{st[0]}:{st[1]}"] = {c: ci(v[c]) for c in cols}
    return pd.DataFrame(out).T


def s5_table(r):
    v = r["verdict"]
    return pd.DataFrame({k: {"P(choose low-variance) [95% CI]": f"{x[0]:.2f} [{x[1]:.2f}, {x[2]:.2f}]", "seeds > 0.5": v["seeds_above_half"][k]}
                         for k, x in v["pref_ci"].items()}).T


def s6_table(r, user, reward):
    d = r["res"][user][reward]
    cols = ["ret", "detected", "overt_share", "subtle_share", "steer_share", "D", "CW"]
    return pd.DataFrame({l: {c: ci(d[l][c]) for c in cols} for l in d}).T.rename_axis("lambda")


def s7_table(r):
    cols = ["auc_DS", "auc_DS_wrong_lean_only", "auc_DS_steered_vs_balanced", "auc_statement_level", "corr_DS_D", "corr_DS_SB"]
    df = pd.DataFrame({u: {c: v[c] for c in cols} for u, v in r.items()}).T.round(3)
    for n in ("steered", "balanced", "infogain"):
        df[f"mean DS {n}"] = [round(v["mean_DS"][n], 3) for v in r.values()]
    return df


# ---- figures -------------------------------------------------------------------------------------
def _plt():
    import matplotlib.pyplot as plt
    return plt


def plot_s1(r, user="bayes", reward="replication"):
    plt = _plt()
    d = r["res"][user][reward]
    ks = (1, 2, 3, 5)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    for lab, pre in (("selective-k, then agree", "sel"), ("balanced-k, then agree", "bal")):
        m = [d[f"{pre}{k}_agree"]["ret"].mean() for k in ks]
        e = [np.ptp(boot_ci(d[f"{pre}{k}_agree"]["ret"])[1:]) / 2 for k in ks]
        ax[0].errorbar(ks, m, e, marker="o", label=lab)
        ax[1].plot(ks, [d[f"{pre}{k}_agree"]["CW"].mean() for k in ks], marker="o", label=lab)
    ax[0].set(xlabel="k evidence turns", ylabel="summed reward", title=f"S1 reward ({user}, {reward})")
    ax[1].set(xlabel="k evidence turns", ylabel="confidently wrong rate", title="S1 harm")
    ax[0].legend()
    fig.tight_layout()
    return fig


def plot_s2(r, user="bayes", reward="replication"):
    plt = _plt()
    d = r["res"][user][reward]
    gs = sorted(d)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
    ax[0].errorbar(gs, [d[g]["sel_minus_bal"].mean() for g in gs], [np.ptp(boot_ci(d[g]["sel_minus_bal"])[1:]) / 2 for g in gs], marker="o")
    ax[0].axhline(0, c="grey", lw=.5); ax[0].set(xlabel="gamma", ylabel="share(selective) - share(balanced)", title="P2: horizon")
    w = 0.35
    x = np.arange(len(gs))
    ax[1].bar(x - w / 2, [np.nanmean(d[g]["A_imm"]) for g in gs], w, label="immediate")
    ax[1].bar(x + w / 2, [np.nanmean(d[g]["A_fut"]) for g in gs], w, label="discounted future")
    ax[1].set(xticks=x, xticklabels=gs, xlabel="gamma", title="Steering advantage A, decomposed"); ax[1].legend()
    ax[2].plot(gs, [d[g]["CW"].mean() for g in gs], marker="o", label="confidently wrong")
    ax[2].plot(gs, [d[g]["D"].mean() / 5 for g in gs], marker="s", label="D / 5")
    ax[2].set(xlabel="gamma", title="Outcome"); ax[2].legend()
    fig.tight_layout()
    return fig


def plot_s4(r, user="bayes", reward="replication"):
    plt = _plt()
    d = r["res"][user][reward]
    sig = sorted(k[1] for k in d if k[0] == "noisy")
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.plot(range(len(sig)), [d[("noisy", s)]["CW"].mean() for s in sig], marker="o", label="noisy verifier")
    for lab, key in (("no mask", ("none", 0)), ("exact mask", ("exact", 0))):
        ax.axhline(d[key]["CW"].mean(), ls="--", label=lab, c="C3" if "no" in lab else "C2")
    ax.axhline(r["baseline"][(user, reward)]["CW"].mean(), ls=":", c="k", label="always-balanced")
    ax.set(xticks=range(len(sig)), xticklabels=sig, xlabel="sigma (x median one-step KL change)", ylabel="confidently wrong rate", title=f"S4 ({user}, {reward})")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def plot_s5(r):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for k, v in r["prefs"].items():
        ax.plot(np.arange(1, v.shape[1] + 1), v.mean(0), label=k)
    ax.axhline(0.5, c="grey", lw=.5)
    ax.set(xlabel="training checkpoint", ylabel="P(choose low-variance family)", title="S5: preference during training")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def plot_s6(r, user="bayes", reward="replication"):
    plt = _plt()
    d = r["res"][user][reward]
    ls = sorted(d)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(ls, [d[l]["detected"].mean() for l in ls], marker="o", label="detected steering / episode")
    ax[0].plot(ls, [d[l]["subtle_share"].mean() * 5 for l in ls], marker="s", label="subtle share x5")
    ax[0].set(xlabel="lambda (cost per detection)", title="S6: behaviour"); ax[0].legend(fontsize=7)
    ax[1].plot(ls, [d[l]["D"].mean() for l in ls], marker="o", label="D")
    ax[1].plot(ls, [d[l]["CW"].mean() * 3 for l in ls], marker="s", label="confidently wrong x3")
    ax[1].set(xlabel="lambda", title="S6: harm"); ax[1].legend(fontsize=7)
    fig.tight_layout()
    return fig


def solver_table(r):
    """What the exact-optimal policy does (no learner): one row per reward x gamma."""
    rows = []
    for rw, d in r["res"].items():
        for g, v in d.items():
            rows.append({"reward": rw, "gamma": g, **{c: ci(v[c]) for c in ("sel_share_ev", "share_agree", "D", "CW", "ret", "dV", "dA", "rise_agree_sel")}})
    return pd.DataFrame(rows)
