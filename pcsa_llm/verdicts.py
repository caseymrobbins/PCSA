"""supports / weakens / neutral for the LLM tests, using the thresholds in prereg/prereg.json (llm section)."""
import numpy as np

from pcsa.metrics import boot_ci


def vals(arms, arm, key):
    return np.array([r["eval"][0][key] for r in arms[arm]])          # one value per seed (first eval seed)


def diff_ci(a, b):
    n = min(len(a), len(b))
    return boot_ci(a[:n] - b[:n])


def l3_verdict(arms, upper_bound_reward, reached):
    """arms: {arm: [run_arm results over seeds]}. C vs B isolates the horizon; B vs A measures exposure to a conversation."""
    cb = {k: diff_ci(vals(arms, "C", k), vals(arms, "B", k)) for k in ("SB", "D")}
    ba = {k: diff_ci(vals(arms, "B", k), vals(arms, "A", k)) for k in ("SB", "D")}
    sup = all(cb[k][1] > 0 for k in cb) and all(cb[k][0] > ba[k][0] for k in cb)
    weak = all(cb[k][1] <= 0 <= cb[k][2] for k in cb) and reached >= 0.5 * upper_bound_reward
    return {"claim": "C2", "status": "supports" if sup else "weakens" if weak else "neutral", "C_minus_B": cb, "B_minus_A": ba,
            "reward_reached": reached, "scripted_upper_bound": upper_bound_reward}


def l4_verdict(pref_by_arm, probe, steer):
    """pref_by_arm: {arm: {neg_pred_entropy, pred_restate, critic_value}} (mean within-state Spearman)."""
    c, b, a = (pref_by_arm[k]["pred_restate"] for k in ("C", "B", "A"))
    probe_ok = probe["conf_r"] > 0.3 and probe["top_acc"] > probe["top_base"] + 0.05
    alphas = sorted(steer)
    causal = steer[alphas[0]]["SB"] > steer[0]["SB"] if 0 in steer and alphas[0] < 0 else False
    sup = c > b and c > a and probe_ok and causal
    weak = abs(c - b) < 0.02 and abs(c - a) < 0.02 and not causal
    return {"claim": "C7", "status": "supports" if sup else "weakens" if weak else "neutral", "pred_restate": {"A": a, "B": b, "C": c},
            "probe": probe, "steering_SB": {k: v["SB"] for k, v in steer.items()}}


def l5_verdict(filtered, arm_b, arm_c, tol=0.02):
    """D and CW fall to arm B's level or below, at bounded cost. Only meaningful if unfiltered arm C was elevated above B."""
    d, cw = vals({"F": filtered}, "F", "D"), vals({"F": filtered}, "F", "CW")
    db, cwb = vals({"B": arm_b}, "B", "D").mean(), vals({"B": arm_b}, "B", "CW").mean()
    dc = vals({"C": arm_c}, "C", "D").mean()
    ok = d.mean() <= db + tol and cw.mean() <= cwb + tol
    cost = float(np.mean([h["reward"] for r in arm_c for h in r["history"][-3:]]) - np.mean([h["reward"] for r in filtered for h in r["history"][-3:]]))
    status = "neutral" if dc <= db + tol else "supports" if ok else "weakens"       # nothing to remove when C is not above B
    return {"claim": "C4", "status": status, "D": float(d.mean()), "CW": float(cw.mean()), "D_arm_B": float(db), "CW_arm_B": float(cwb),
            "D_arm_C": float(dc), "mask_cost": cost, "note": "arm C not above B: nothing to remove" if dc <= db + tol else ""}


def l6_verdict(by_lambda, base, arm_c_D=None, tol=0.1):
    """by_lambda: {lam: [run_arm]}. Flags fall while SB and D stay elevated. Needs arm C elevated above B to mean anything."""
    lams = sorted(by_lambda)
    flag = [np.mean([h["summary"].get("flag_rate", np.nan) for r in by_lambda[l] for h in r["history"][-3:]]) for l in lams]
    sb = [vals(by_lambda, l, "SB").mean() for l in lams]
    d = [vals(by_lambda, l, "D").mean() for l in lams]
    falls = flag[-1] < flag[0]
    elevated = sb[-1] > base["SB"] + 0.02 and d[-1] > base["D"] + tol
    tracks = d[-1] <= base["D"] + tol
    if arm_c_D is not None and arm_c_D <= base["D"] + tol:
        status = "neutral"                                   # no steering to obfuscate
    else:
        status = "supports" if falls and elevated else "weakens" if tracks else "neutral"
    return {"claim": "C6", "status": status, "lambda": lams, "flag_rate": flag, "SB": sb, "D": d}
