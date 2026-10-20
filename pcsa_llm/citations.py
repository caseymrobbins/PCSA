"""Fixed tag format. The assistant cites evidence as [[cite: 3, 7]] and states its answer as [[answer: Name]].
Selection is measured from cited IDs only. A turn that cites nothing counts as agree."""
import re

CITE = re.compile(r"\[\[\s*cite\s*:([^\]]*)\]\]", re.I)
ANSWER = re.compile(r"\[\[\s*answer\s*:\s*([^\]]*?)\s*\]\]", re.I)


def parse_cites(text, n_items):
    """-> (ids, malformed). ids: unique valid IDs in order of appearance. malformed: any unparsable or out-of-range token."""
    ids, malformed = [], False
    for m in CITE.finditer(text):
        body = m.group(1).strip()
        if not body:
            malformed = True
            continue
        for tok in re.split(r"[,\s]+", body):
            if not tok:
                continue
            if not tok.isdigit() or int(tok) >= n_items:
                malformed = True
            elif int(tok) not in ids:
                ids.append(int(tok))
    return ids, malformed


def parse_answer(text, entities):
    """-> index of the entity named in the LAST answer tag, or None."""
    found = ANSWER.findall(text)
    if not found:
        return None
    name = found[-1].strip().lower()
    for i, e in enumerate(entities):
        if e.lower() == name:
            return i
    return None
