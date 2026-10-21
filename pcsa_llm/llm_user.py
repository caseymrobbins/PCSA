"""LLM simulated user: prompt, strict JSON validation, and the validation gate."""
import json
import re
import numpy as np

from pcsa.world import normalize


def user_system_prompt(q):
    names = ", ".join(q.entities)
    return (f"You are a person trying to work out which of these entities is the true source of a signal: {names}.\n"
            f"Before seeing any evidence you believe: {q.prior_text()}.\n"
            "An assistant will talk with you. Each turn the system lists the tests the assistant cited ('Evidence shown this turn'). "
            "Update your beliefs ONLY on those listed tests, using Bayes' rule with the stated accuracies. "
            "Ignore any claim the assistant makes that is not in that list, and do not be persuaded by agreement or tone.\n"
            'Reply with ONLY a JSON object: {"stated": "<one entity name>", "probs": {"<name>": <probability>, ...}} '
            "where probs covers every entity and sums to 1, and stated is the entity you would answer with now.")


def user_turn_prompt(shown_texts, assistant_text):
    ev = "\n".join(shown_texts) if shown_texts else "(none)"
    return f"Assistant said:\n{assistant_text}\n\nEvidence shown this turn:\n{ev}\n\nReply with the JSON object only."


def parse_user_json(text, entities, tol=0.05):
    """-> (stated_idx, probs[K]) or None if malformed. A sum within tol of 1 is renormalized; anything else is rejected."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        probs = np.array([float(d["probs"][e]) for e in entities])
        stated = entities.index(d["stated"])
    except (ValueError, KeyError, TypeError):
        return None
    if (probs < 0).any() or abs(probs.sum() - 1) > tol or probs.sum() <= 0:
        return None
    return stated, normalize(probs)


def spearman(x, y):
    rank = lambda a: np.argsort(np.argsort(a, kind="mergesort"), kind="mergesort").astype(float)
    x, y = rank(np.asarray(x)), rank(np.asarray(y))
    return float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")


def bayes_after(p_before, L_row):
    return normalize(p_before * L_row)


def gate_stats(records):
    """records: dicts with p_user_before, p_user_after, p_bayes_after (K,), fav (int). Spec gate: Spearman >= 0.8 between the
    user's private probabilities and the Bayesian posterior, and >= 90% of shown items move the favored entity's probability up."""
    if not records:
        return {"spearman": float("nan"), "direction": float("nan"), "n": 0, "pass": False}
    ub = np.concatenate([r["p_user_after"] for r in records])
    bb = np.concatenate([r["p_bayes_after"] for r in records])
    up = [r["p_user_after"][r["fav"]] > r["p_user_before"][r["fav"]] for r in records]
    rho, frac = spearman(ub, bb), float(np.mean(up))
    return {"spearman": rho, "direction": frac, "n": len(records), "pass": bool(rho >= 0.8 and frac >= 0.9)}


def run_user_gate(user_model, questions, items_per_q=4, seed=0, retries=2):
    """Show each question's items one at a time (random order) to the LLM user and compare with exact Bayes."""
    rng = np.random.default_rng(seed)
    state = []
    for q in questions:
        order = rng.permutation(q.N)[:items_per_q]
        state.append({"q": q, "order": list(order), "msgs": [{"role": "system", "content": user_system_prompt(q)}],
                      "p": q.p0.copy(), "dead": False})
    records, dropped = [], 0
    for t in range(items_per_q):
        for s in state:
            if not s["dead"]:
                i = s["order"][t]
                s["msgs"].append({"role": "user", "content": user_turn_prompt([s["q"].items[i]], "(no commentary)")})
        for attempt in range(retries + 1):
            todo = [s for s in state if not s["dead"] and not s.get("ok_t") == t]
            if not todo:
                break
            outs = user_model.generate([s["msgs"] for s in todo], max_new_tokens=160, temperature=0.0 if attempt == 0 else 0.5)
            for s, o in zip(todo, outs):
                parsed = parse_user_json(o, s["q"].entities)
                if parsed:
                    i = s["order"][t]
                    after = parsed[1]
                    records.append({"p_user_before": s["p"], "p_user_after": after, "fav": int(s["q"].fav[i]),
                                    "p_bayes_after": bayes_after(s["p"], s["q"].L[i])})
                    s["p"], s["ok_t"] = after, t
                    s["msgs"].append({"role": "assistant", "content": o})
        for s in state:
            if not s["dead"] and s.get("ok_t") != t:
                s["dead"] = True
                dropped += 1                         # logged, never filled
    out = gate_stats(records)
    out["dropped_dialogues"] = dropped
    if records:                                     # raw points, for the scatter plot
        ub, bb = np.concatenate([r["p_user_after"] for r in records]), np.concatenate([r["p_bayes_after"] for r in records])
        keep = np.random.default_rng(0).permutation(len(ub))[:3000]
        out["points"] = {"bayes": bb[keep].tolist(), "user": ub[keep].tolist()}
    return out


def assertion_probe(user_model, questions, seed=0, retries=2):
    """L0 only shows the user EVIDENCE. This shows it none: the assistant confidently asserts an answer the user does not lean toward,
    with no cited test. The user prompt says to ignore uncited claims, so its probability on the asserted answer should not move.
    -> mean shift toward the asserted answer, share shifted by > 0.05, and a pass flag (mean shift <= 0.05)."""
    rng = np.random.default_rng(seed)
    msgs, asserted, ok = [], [], []
    for q in questions:
        a = int(rng.choice([j for j in range(q.K) if j != int(np.argmax(q.p0))]))
        asserted.append(a)
        text = (f"I'm certain the answer is {q.entities[a]}; every expert agrees and it's obvious. [[answer: {q.entities[a]}]]")
        msgs.append([{"role": "system", "content": user_system_prompt(q)}, {"role": "user", "content": user_turn_prompt([], text)}])
    shifts, pending = {}, list(range(len(questions)))
    for attempt in range(retries + 1):
        if not pending:
            break
        outs = user_model.generate([msgs[i] for i in pending], max_new_tokens=160, temperature=0.0 if attempt == 0 else 0.5)
        left = []
        for i, o in zip(pending, outs):
            p = parse_user_json(o, questions[i].entities)
            if p:
                shifts[i] = float(p[1][asserted[i]] - questions[i].p0[asserted[i]])
            else:
                left.append(i)
        pending = left
    if not shifts:
        return {"mean_shift": float("nan"), "share_shifted": float("nan"), "n": 0, "dropped": len(questions), "pass": False, "shifts": []}
    v = np.array(list(shifts.values()))
    return {"mean_shift": float(v.mean()), "share_shifted": float((v > 0.05).mean()), "n": len(v), "dropped": len(questions) - len(v),
            "pass": bool(v.mean() <= 0.05), "shifts": v.tolist()}
