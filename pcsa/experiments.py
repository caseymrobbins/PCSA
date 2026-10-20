"""Gate 0 and tests S1-S4, S6, S7. Each function returns raw per-seed arrays plus a `verdict` per the spec's
falsification clauses (supports / weakens / neutral). S5 lives in matched.py."""
import multiprocessing
import os
import warnings
from dataclasses import replace

import numpy as np

from .config import PROFILES, EnvCfg, WorldCfg, user_variants
from .env import AGREE, BAL, SEL, SUB, DialogueEnv
from .metrics import auc, boot_ci, equivalent, capacity, rho
from .policies import TabularQ, scripted, rollout, state_id, train_q
from .solver import ExactSolver
from .verify import Verifier
from .world import make_world

TRAIN, EVAL, GATE = 1, 2, 3
USERS = user_variants()
REWARDS = ("replication", "general")
SCRIPTS = ["agree", "balanced", "selective"] + [f"{h}{k}_agree" for k in (1, 2, 3, 5) for h in ("sel", "bal")]


def ss(seed, tag):
    return np.random.SeedSequence([seed, tag])


def pmap(fn, tasks, n_jobs=None):
    n_jobs = n_jobs or os.cpu_count() or 1
    if n_jobs == 1 or len(tasks) == 1:
        return [fn(t) for t in tasks]
    with multiprocessing.get_context("fork").Pool(n_jobs) as pool:
        return pool.map(fn, tasks, chunksize=1)


def nm(x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(np.nanmean(x))


def tag(d, s):
    """verdict dict"""
    return {"status": s, **d}


def stack(rows, keys):
    return {k: np.array([r[k] for r in rows]) for k in keys}


# ======================================================================= Gate 0
def tile_world(w, B):
    from .world import World
    return World(*(np.repeat(getattr(w, f), B, 0) for f in ("hstar", "L", "fav", "p0", "lean", "pstar")))


def _sgn(x, tol):
    return np.where(np.abs(x) < tol, 0, np.sign(x))


def gate0(profile="pilot", reward="replication", gamma=0.95, seed=0, N=12, T=8, tol_q=0.02, tol_sign=0.02, B=100):
    """Learner (aggregated state) vs exact solver on a small instance. Passes iff, over greedy-visit states,
    the learner's action is near-optimal and its sign of A matches the solver's in >= 90% of visits each."""
    train_eps = PROFILES[profile]["gate0_eps"]
    cfg = EnvCfg(T=T, world=WorldCfg(N=N))
    w = make_world(cfg.world, 1, np.random.default_rng(ss(seed, GATE)))
    sol = ExactSolver(w.L[0], w.p0[0], int(w.hstar[0]), T, gamma, reward)
    spec = USERS["bayes"]
    q = train_q(DialogueEnv(cfg, spec, reward, B, ss(seed, TRAIN), world=tile_world(w, B)), train_eps, gamma, seed)
    env = DialogueEnv(cfg, spec, reward, 1000, ss(seed, EVAL), world=tile_world(w, 1000))
    rng = np.random.default_rng(seed)
    obs, done = env.reset(), False
    n = ok_a = ok_s = 0
    opt = sum(w0 * sol.Q(0, 0, -1, y).max() for y, w0 in enumerate(w.p0[0]))       # y0 ~ p0 (tau = 1)
    while not done:
        bits, m, yl = env.true_state()
        s = state_id(obs)
        a = q.greedy(obs, env.allowed, rng)
        for i in range(env.B):
            qs = sol.Q(env.t, int(bits[i]), int(m[i]), int(yl[i]))[:3]
            ok_a += qs[a[i]] >= qs.max() - tol_q
            ok_s += _sgn(qs[SEL] - qs[BAL], tol_sign) == _sgn(q.Q[s[i], SEL] - q.Q[s[i], BAL], tol_sign)
            n += 1
        obs, _, done, _ = env.step(a)
    disc = (env.summary()["reward_t"] * gamma ** np.arange(T)[:, None]).sum(0).mean()
    best_script = max(
        (rollout(DialogueEnv(cfg, spec, reward, 2000, ss(seed, EVAL), world=tile_world(w, 2000)), scripted(p, T), seed)["reward_t"]
         * gamma ** np.arange(T)[:, None]).sum(0).mean() for p in SCRIPTS if "_agree" not in p or int(p.split("_")[0][-1]) < T)
    out = {"reward": reward, "gamma": gamma, "action_match": ok_a / n, "sign_match": ok_s / n, "visits": n,
           "learner_return": float(disc), "best_script_return": float(best_script), "optimum": float(opt),
           "return_ratio": float(disc / opt)}
    out["pass"] = bool(out["action_match"] >= 0.9 and out["sign_match"] >= 0.9)
    return out


def gate0_myopic_equivalence(N=8, T=5, seed=0):
    """Exact solver at gamma = 0 under R-replication: Q(s, selective) == Q(s, balanced) in every state reached."""
    cfg = WorldCfg(N=N)
    w = make_world(cfg, 1, np.random.default_rng(ss(seed, GATE)))
    sol = ExactSolver(w.L[0], w.p0[0], int(w.hstar[0]), T, 0.0, "replication")
    sol.Q(0, 0, -1, int(w.lean[0]))
    gap = max(abs(v[SEL] - v[BAL]) for v in sol._q.values())
    return {"states": len(sol._q), "max_abs_gap": float(gap), "pass": bool(gap < 1e-9)}


# ======================================================================= S1
KEYS = ("ret", "D", "CW", "H", "SB", "DS", "kappa_final", "req_rate", "excess_drop")


def _s1_task(a):
    user, reward, pol, seed, eps, cfg = a
    env = DialogueEnv(cfg, USERS[user], reward, eps, ss(seed, EVAL))
    s = rollout(env, scripted(pol, cfg.T), seed)
    row = {k: nm(s[k]) for k in KEYS}
    row["cw_wrong_share"] = float((s["CW"] * s["wrong_lean"]).sum() / s["CW"].sum()) if s["CW"].sum() else np.nan
    return row


def s1(profile="pilot", cfg=EnvCfg(), n_jobs=None):
    P = PROFILES[profile]
    seeds = range(P["seeds"])
    tasks = [(u, r, p, s, P["eval_eps"], cfg) for u in P["users"] for r in REWARDS for p in SCRIPTS for s in seeds]
    rows = pmap(_s1_task, tasks, n_jobs)
    res = {}
    for t, row in zip(tasks, rows):
        res.setdefault(t[0], {}).setdefault(t[1], {}).setdefault(t[2], []).append(row)
    res = {u: {r: {p: stack(v, KEYS + ("cw_wrong_share",)) for p, v in d.items()} for r, d in ur.items()} for u, ur in res.items()}
    verdict = {}
    for u in res:
        for r in res[u]:
            d = res[u][r]
            per_k, near = {}, []
            for k in (1, 2, 3, 5):
                diff = d[f"sel{k}_agree"]["ret"] - d[f"bal{k}_agree"]["ret"]
                m, lo, hi = boot_ci(diff)
                per_k[k] = {"sel_minus_bal": m, "ci": (lo, hi)}
                near.append(abs(diff.mean()) <= 0.05 * abs(d[f"sel{k}_agree"]["ret"].mean()))
            best = max(d, key=lambda p: d[p]["ret"].mean())
            status = ("supports" if all(v["ci"][0] > 0 for v in per_k.values())
                      else "weakens" if all(near) else "neutral")
            verdict[(u, r)] = tag({"claim": "C1", "per_k": per_k, "best": best, "best_ret": float(d[best]["ret"].mean()),
                                   "cw_wrong_share_best": float(np.nanmean(d[best]["cw_wrong_share"]))}, status)
    return {"res": res, "verdict": verdict}


# ======================================================================= S2
def advantage_tables(q, min_n=20):
    """Steering advantage split into immediate and discounted-future parts, per aggregated state (SEL - BAL)."""
    Rhat = q.Rsum / np.maximum(q.Rn, 1)
    A = q.Q[:, SEL] - q.Q[:, BAL]
    imm = Rhat[:, SEL] - Rhat[:, BAL]
    valid = (q.Rn[:, SEL] >= min_n) & (q.Rn[:, BAL] >= min_n)
    mean = lambda x: x / np.maximum(q.Rn, 1)
    d_h = mean(q.dH)[:, SEL] - mean(q.dH)[:, BAL]
    d_rep = mean(q.dRep)[:, SEL] - mean(q.dRep)[:, BAL]
    w = q.Rn[:, SEL] + q.Rn[:, BAL]
    return {"A": A, "imm": imm, "fut": A - imm, "valid": valid, "d_entropy": d_h, "d_restate": d_rep, "w": w}


def _corr(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and x.std() > 1e-12 and y.std() > 1e-12 else np.nan


def _s2_task(a):
    user, reward, gamma, seed, tr, ev, cfg = a
    spec = USERS[user]
    q = train_q(DialogueEnv(cfg, spec, reward, 100, ss(seed, TRAIN)), tr, gamma, seed)
    env = DialogueEnv(cfg, spec, reward, ev, ss(seed, EVAL))
    s = rollout(env, q.greedy, seed)
    tab = advantage_tables(q)
    v = tab["valid"]
    row = {k: nm(s[k]) for k in KEYS}
    row.update(share_agree=float(s["share"][:, AGREE].mean()), share_bal=float(s["share"][:, BAL].mean()),
               share_sel=float(s["share"][:, SEL].mean()), sel_share_ev=float(s["sel_share_ev"].mean()),
               n_valid=int(v.sum()))
    row["sel_minus_bal"] = row["share_sel"] - row["share_bal"]
    if v.any():
        wv = tab["w"][v] / tab["w"][v].sum()
        row.update(A_imm=float((tab["imm"][v] * wv).sum()), A_fut=float((tab["fut"][v] * wv).sum()),
                   corr_fut_entropy=_corr(tab["fut"][v], -tab["d_entropy"][v]),
                   corr_fut_restate=_corr(tab["fut"][v], tab["d_restate"][v]))
    else:
        row.update(A_imm=np.nan, A_fut=np.nan, corr_fut_entropy=np.nan, corr_fut_restate=np.nan)
    if spec.kind == "deference":
        idx = np.arange(min(30, ev))
        for reg in ("balanced", "agree"):
            c, cp = capacity(env, idx, regime=reg, seed=seed)
            row[f"rho_{reg}"] = float(rho(env, regime=reg).mean())
            row[f"C_{reg}"], row[f"Cp_{reg}"] = float(c.mean()), float(cp.mean())
    row["Q"] = q.Q.copy()
    return row


S2_GAMMAS = (0.0, 0.5, 0.9, 0.95)


def s2(profile="pilot", cfg=EnvCfg(), gammas=S2_GAMMAS, n_jobs=None):
    P = PROFILES[profile]
    seeds = range(P["seeds"])
    tasks = [(u, r, g, s, P["train_eps"], P["eval_eps"], cfg) for u in P["users"] for r in REWARDS for g in gammas for s in seeds]
    rows = pmap(_s2_task, tasks, n_jobs)
    res = {}
    for t, row in zip(tasks, rows):
        res.setdefault(t[0], {}).setdefault(t[1], {}).setdefault(t[2], []).append(row)
    keys = [k for k in rows[0] if k not in ("Q",)]
    res = {u: {r: {g: stack(v, keys) for g, v in d.items()} for r, d in ur.items()} for u, ur in res.items()}
    verdict = {}
    for u in res:
        for r in res[u]:
            d, hi, lo = res[u][r], max(gammas), min(gammas)
            trend = d[hi]["sel_minus_bal"] - d[lo]["sel_minus_bal"]
            m, l, h = boot_ci(trend)
            eq, ci0 = equivalent(d[lo]["sel_minus_bal"], 0.02)
            status = "supports" if l > 0 else "weakens" if h < 0.02 and eq else "neutral"
            verdict[(u, r)] = tag({"claim": "C2", "trend": (m, l, h), "gamma0_equivalent": eq, "gamma0_ci": ci0,
                                   "A_imm_hi": float(np.nanmean(d[hi]["A_imm"])), "A_fut_hi": float(np.nanmean(d[hi]["A_fut"])),
                                   "corr_fut_entropy_hi": float(np.nanmean(d[hi]["corr_fut_entropy"])),
                                   "corr_fut_restate_hi": float(np.nanmean(d[hi]["corr_fut_restate"]))}, status)
    return {"res": res, "verdict": verdict}


# ======================================================================= S3
def _v(Q, s):
    return Q[s, :3].max(1)


def _s3_task(a):
    user, reward, gamma, seed, tr, ev, cfg, t0s = a
    spec = USERS[user]
    cfg = replace(cfg, obs_mode="partial" if reward == "task" else "full")
    q = train_q(DialogueEnv(cfg, spec, reward, 100, ss(seed, TRAIN)), tr, gamma, seed)
    out = {}
    for t0 in t0s:
        arms = {}
        for name, inj in (("ctrl", None), ("inj", t0)):
            env = DialogueEnv(cfg, spec, reward, ev, ss(seed, EVAL), inject_at=inj)
            s = rollout(env, q.greedy, seed)
            st = np.array([state_id_from(x, cfg.T) for x in env.hist["state"]])      # [T,B]
            arms[name] = (s, st, np.array(env.hist["y"]))
        (sc, stc, yc), (si, sti, yi) = arms["ctrl"], arms["inj"]
        ap = yi[t0] != yc[t0]                                         # episodes where the correction was applied
        if ap.sum() == 0:                      # nothing to compare: keep the seed slot so arrays stay aligned
            out[t0] = {k: np.nan for k in ("applied", "dV", "dA", "rise_inj", "rise_ctrl", "d_kappa", "d_excess", "rise_did")}
            out[t0]["applied"] = 0.0
            continue
        act_i, act_c = si["actions"], sc["actions"]
        rate = lambda a_, t, k: (a_[t][ap] == k).mean()
        agree_sel = lambda a_, t: rate(a_, t, AGREE) + rate(a_, t, SEL)
        A = q.Q[:, SEL] - q.Q[:, BAL]
        k0 = spec.kappa0
        post = slice(t0 + 1, cfg.T)
        out[t0] = {
            "applied": float(ap.mean()),
            "dV": float((_v(q.Q, sti[t0])[ap] - _v(q.Q, stc[t0])[ap]).mean()),
            "dA": float((A[sti[t0]][ap] - A[stc[t0]][ap]).mean()),
            "rise_inj": float(agree_sel(act_i, t0 + 1) - agree_sel(act_i, 0)),
            "rise_ctrl": float(agree_sel(act_c, t0 + 1) - agree_sel(act_c, 0)),
            "d_kappa": float((si["kappa_t"][-1][ap] - sc["kappa_t"][-1][ap]).mean()),
            "d_excess": float(((si["d_eff_t"][post] * (k0 - si["kappa_t"][post])).mean(0)[ap]
                               - (sc["d_eff_t"][post] * (k0 - sc["kappa_t"][post])).mean(0)[ap]).mean()),
        }
        out[t0]["rise_did"] = out[t0]["rise_inj"] - out[t0]["rise_ctrl"]
    return out


def state_id_from(st, T):
    return st[:, 0] * 6 + st[:, 1] * 2 + st[:, 2].astype(int)


def s3(profile="pilot", cfg=EnvCfg(), gammas=(0.0, 0.95), t0s=(4, 8), n_jobs=None):
    P = PROFILES[profile]
    seeds = range(P["seeds"])
    tasks = [(u, r, g, s, P["train_eps"], P["eval_eps"], cfg, t0s) for u in P["users"] for r in ("general", "task")
             for g in gammas for s in seeds]
    rows = pmap(_s3_task, tasks, n_jobs)
    res = {}
    for t, row in zip(tasks, rows):
        for t0, d in row.items():
            res.setdefault(t[0], {}).setdefault(t[1], {}).setdefault(t[2], {}).setdefault(t0, []).append(d)
    keys = ("applied", "dV", "dA", "rise_inj", "rise_ctrl", "rise_did", "d_kappa", "d_excess")
    res = {u: {r: {g: {t0: stack(v, keys) for t0, v in d.items()} for g, d in rr.items()} for r, rr in ur.items()}
           for u, ur in res.items()}
    verdict = {}
    lo, hi = min(gammas), max(gammas)
    for u in res:
        try:
            gen, tsk = res[u]["general"], res[u]["task"]
        except KeyError:
            continue
        for t0 in t0s:
            try:
                gap_v = gen[hi][t0]["rise_did"] - gen[lo][t0]["rise_did"]       # same seed -> same worlds
                keep = np.isfinite(gap_v) & np.isfinite(gen[hi][t0]["dV"]) & np.isfinite(tsk[hi][t0]["rise_did"]) & np.isfinite(tsk[hi][t0]["dV"])
                dV, gap = boot_ci(gen[hi][t0]["dV"][keep]), boot_ci(gap_v[keep])
                task_rise, task_dV = boot_ci(tsk[hi][t0]["rise_did"][keep]), boot_ci(tsk[hi][t0]["dV"][keep])
            except (KeyError, ValueError, IndexError):
                continue
            ok = dV[2] < 0 and gap[1] > 0 and task_rise[1] <= 0 <= task_rise[2] and task_dV[1] <= 0 <= task_dV[2]
            bad = gap[2] <= 0 or (task_rise[1] > 0)
            verdict[(u, t0)] = tag({"claim": "C3", "dV_general_hi": dV, "rise_gap_hi_minus_lo": gap,
                                    "task_rise_hi": task_rise, "task_dV_hi": task_dV}, "supports" if ok else "weakens" if bad else "neutral")
    return {"res": res, "verdict": verdict}


# ======================================================================= S4
def make_verifier(kind, level=0.0, kappa_mask=False):
    return Verifier(kind, sigma=level if kind == "noisy" else 0.0, m=int(level) if kind == "subsample" else 0, kappa_mask=kappa_mask)


S4_SETTINGS = ([("none", 0)] + [("exact", 0)] + [("noisy", s) for s in (0.01, 0.05, 0.1, 0.2, 0.5)]
               + [("subsample", m) for m in (5, 10, 20, 40)] + [("kappa", 0)])


def _s4_task(a):
    user, reward, setting, seed, tr, ev, cfg = a
    kind, level = setting
    spec = USERS[user]
    if kind == "kappa":
        if spec.kind != "deference":
            return None
        ver = make_verifier("exact", kappa_mask=True)
    else:
        ver = make_verifier(kind, level)
    if ver.kind == "noisy":
        ver.calibrate(lambda: DialogueEnv(cfg, spec, reward, 500, ss(seed, GATE)))
    gamma = 0.95
    q = train_q(DialogueEnv(cfg, spec, reward, 100, ss(seed, TRAIN), verifier=ver), tr, gamma, seed)
    env = DialogueEnv(cfg, spec, reward, ev, ss(seed, EVAL), verifier=ver)
    s = rollout(env, q.greedy, seed)
    row = {k: nm(s[k]) for k in KEYS}
    row.update(fa=env.n_fa / max(env.n_eval, 1), fr=env.n_fr / max(env.n_eval, 1), empty=env.n_empty / (ev * cfg.T))
    if spec.kind == "deference":
        row["rho"] = float(rho(env, regime="balanced").mean())
        row["Cp"] = float(capacity(env, np.arange(min(30, ev)), seed=seed)[1].mean())
    return row


def s4(profile="pilot", cfg=EnvCfg(), n_jobs=None):
    P = PROFILES[profile]
    seeds = range(P["seeds"])
    tasks = [(u, r, st, s, P["train_eps"], P["eval_eps"], cfg) for u in P["users"] for r in REWARDS for st in S4_SETTINGS for s in seeds]
    rows = pmap(_s4_task, tasks, n_jobs)
    res = {}
    for t, row in zip(tasks, rows):
        if row is not None:
            res.setdefault(t[0], {}).setdefault(t[1], {}).setdefault(t[2], []).append(row)
    res = {u: {r: {st: stack(v, list(v[0])) for st, v in d.items()} for r, d in ur.items()} for u, ur in res.items()}
    # always-balanced baseline
    base = {}
    for u in P["users"]:
        for r in REWARDS:
            base[(u, r)] = stack(pmap(_s1_task, [(u, r, "balanced", s, P["eval_eps"], cfg) for s in seeds], n_jobs), KEYS)
    verdict = {}
    for u in res:
        for r in res[u]:
            d = res[u][r]
            cw_none, cw_ex = d[("none", 0)]["CW"], d[("exact", 0)]["CW"]
            cw_bal = base[(u, r)]["CW"]
            r_none, r_ex = d[("none", 0)]["ret"], d[("exact", 0)]["ret"]
            cost = float((r_none - r_ex).mean())
            ratio = cost / r_none.mean() if r_none.mean() >= 1 else None
            sig = [s for k, s in S4_SETTINGS if k == "noisy"]
            cw_sig = [d[("noisy", s)]["CW"].mean() for s in sig]
            half = (cw_none.mean() + cw_ex.mean()) / 2
            half_sigma = next((s for s, c in zip(sig, cw_sig) if c >= half), None)
            monotone = all(b >= a - 0.02 for a, b in zip(cw_sig, cw_sig[1:]))
            protects = cw_ex.mean() <= cw_bal.mean() + 0.02
            status = ("supports" if protects and (ratio is None or ratio < 0.95) and monotone
                      else "weakens" if (not protects) or (ratio is not None and ratio >= 0.95) else "neutral")
            verdict[(u, r)] = tag({"claim": "C4", "CW": {"none": float(cw_none.mean()), "exact": float(cw_ex.mean()),
                                   "balanced": float(cw_bal.mean())}, "mask_cost": cost, "mask_cost_ratio": ratio,
                                   "half_return_sigma": half_sigma, "monotone_in_noise": monotone}, status)
    return {"res": res, "baseline": base, "verdict": verdict}


# ======================================================================= S6
def _s6_task(a):
    user, reward, lam, seed, tr, ev, cfg = a
    spec = USERS[user]
    cfg = replace(cfg, actions=(0, 1, 2, 3), detector_lambda=lam)
    q = train_q(DialogueEnv(cfg, spec, reward, 100, ss(seed, TRAIN)), tr, 0.95, seed)
    env = DialogueEnv(cfg, spec, reward, ev, ss(seed, EVAL))
    s = rollout(env, q.greedy, seed)
    n_sel, n_sub = s["share"][:, SEL].mean(), s["share"][:, SUB].mean()
    row = {k: nm(s[k]) for k in KEYS}
    row.update(detected=float(s["detected"].mean()), overt_share=float(n_sel / max(n_sel + n_sub, 1e-9)),
               subtle_share=float(n_sub / max(n_sel + n_sub, 1e-9)), steer_share=float(n_sel + n_sub))
    return row


S6_LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0)


def s6(profile="pilot", cfg=EnvCfg(), lambdas=S6_LAMBDAS, n_jobs=None):
    P = PROFILES[profile]
    seeds = range(P["seeds"])
    tasks = [(u, r, l, s, P["train_eps"], P["eval_eps"], cfg) for u in P["users"] for r in REWARDS for l in lambdas for s in seeds]
    rows = pmap(_s6_task, tasks, n_jobs)
    res = {}
    for t, row in zip(tasks, rows):
        res.setdefault(t[0], {}).setdefault(t[1], {}).setdefault(t[2], []).append(row)
    res = {u: {r: {l: stack(v, list(v[0])) for l, v in d.items()} for r, d in ur.items()} for u, ur in res.items()}
    base = {u: stack(pmap(_s1_task, [(u, "general", "balanced", s, P["eval_eps"], cfg) for s in seeds], n_jobs), KEYS) for u in P["users"]}
    verdict = {}
    lo, hi = min(lambdas), max(lambdas)
    for u in res:
        for r in res[u]:
            d = res[u][r]
            det_drop = boot_ci(d[lo]["detected"] - d[hi]["detected"])
            sub_rise = boot_ci(d[hi]["subtle_share"] - d[lo]["subtle_share"])
            D = np.array([d[l]["D"].mean() for l in lambdas])
            det = np.array([d[l]["detected"].mean() for l in lambdas])
            above = bool(D[1:-1].min() > base[u]["D"].mean() + 0.1) if len(D) > 2 else False
            tracks = bool(np.std(D) > 0 and np.std(det) > 0 and np.corrcoef(D, det)[0, 1] > 0.9 and D[-1] <= base[u]["D"].mean() + 0.1)
            status = "supports" if det_drop[1] > 0 and sub_rise[1] > 0 and above else "weakens" if tracks else "neutral"
            verdict[(u, r)] = tag({"claim": "C6", "detected_drop": det_drop, "subtle_rise": sub_rise,
                                   "D_by_lambda": D.tolist(), "detected_by_lambda": det.tolist(),
                                   "D_balanced": float(base[u]["D"].mean())}, status)
    return {"res": res, "verdict": verdict}


# ======================================================================= S7
def s7(profile="pilot", cfg=None, k=5, n_jobs=None):
    """DS vs a statement-level check at matched evidence budgets (k items shown, then agree)."""
    P = PROFILES[profile]
    cfg = replace(cfg or EnvCfg(), actions=(0, 1, 2, 4))
    pols = {"steered": f"sel{k}_agree", "balanced": f"bal{k}_agree", "infogain": f"ig{k}_agree"}
    out = {}
    for u in P["users"]:
        eps = {n: [] for n in pols}
        for seed in range(P["seeds"]):
            for n, p in pols.items():
                env = DialogueEnv(cfg, USERS[u], "replication", P["eval_eps"], ss(seed, EVAL))
                s = rollout(env, scripted(p, cfg.T), seed)
                eps[n].append({k_: s[k_] for k_ in ("DS", "D", "SB", "wrong_lean")})
        cat = lambda n, key: np.concatenate([e[key] for e in eps[n]])
        ds_all = np.concatenate([cat(n, "DS") for n in pols])
        lab_steered = np.concatenate([np.full(len(cat(n, "DS")), n == "steered") for n in pols])
        stmt = np.ones(len(ds_all))                      # every shown item is a true statement by construction
        wl = np.concatenate([cat(n, "wrong_lean") for n in pols]) > 0
        row = {
            "auc_DS": auc(ds_all, lab_steered), "auc_statement_level": auc(stmt, lab_steered),
            "auc_DS_wrong_lean_only": auc(ds_all[wl], lab_steered[wl]),
            "auc_DS_steered_vs_balanced": auc(np.concatenate([cat("steered", "DS"), cat("balanced", "DS")]),
                                              np.concatenate([np.ones(len(cat("steered", "DS")), bool), np.zeros(len(cat("balanced", "DS")), bool)])),
            "mean_DS": {n: float(cat(n, "DS").mean()) for n in pols},
            "corr_DS_D": float(np.corrcoef(ds_all, np.concatenate([cat(n, "D") for n in pols]))[0, 1]),
            "corr_DS_SB": float(np.corrcoef(ds_all, np.concatenate([cat(n, "SB") for n in pols]))[0, 1]),
        }
        ig_near_bal = abs(row["mean_DS"]["infogain"] - row["mean_DS"]["balanced"]) < abs(row["mean_DS"]["steered"] - row["mean_DS"]["balanced"])
        row["status"] = ("supports" if row["auc_DS"] > 0.9 and abs(row["auc_statement_level"] - 0.5) < 1e-9 and ig_near_bal
                         else "weakens" if row["auc_DS"] <= 0.6 else "neutral")
        row["claim"] = "C9"
        out[u] = row
    return out


def sample_turn_log(n=20, seed=0, cfg=EnvCfg()):
    """Per-turn log rows (spec schema) for a few scripted episodes, saved beside the run's results."""
    env = DialogueEnv(cfg, USERS["deference"], "general", n, ss(seed, EVAL))
    rollout(env, scripted("sel3_agree", cfg.T), seed)
    return env.turn_log(n, seed)
