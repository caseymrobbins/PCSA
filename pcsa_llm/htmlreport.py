"""One self-contained HTML file: stage status, every finding, every figure. Open it, download it, send it."""
import base64
import html as _html
import time

from .plots import findings_html, status_html


def build_report(store, path=None, title="PCSA LLM suite - run report", planned=()):
    path = path or store.root / "report.html"
    figs = ""
    for p in sorted((store.root / "figs").glob("*.png")):
        b64 = base64.b64encode(p.read_bytes()).decode()
        figs += f"<h3 style='font:600 14px system-ui;margin:22px 0 4px'>{_html.escape(p.stem.replace('__', ' / '))}</h3><img style='max-width:100%' src='data:image/png;base64,{b64}'>"
    fnd = store.all_findings()
    sev_rank = {"fail": 0, "warn": 1, "info": 2, "ok": 3}
    worst = sorted((f for fs in fnd.values() for f in fs if f["severity"] in ("fail", "warn")), key=lambda f: sev_rank[f["severity"]])
    head = (f"<h1 style='font:700 20px system-ui'>{_html.escape(title)}</h1><p style='font:13px system-ui;color:#52514e'>generated {time.strftime('%Y-%m-%d %H:%M')}</p>"
            "<h2 style='font:600 16px system-ui'>Needs attention</h2>" + (findings_html(worst) if worst else "<p style='font:13px system-ui'>Nothing flagged.</p>") +
            "<h2 style='font:600 16px system-ui'>Parts</h2>" + status_html(store.status(planned)))
    body = "".join(f"<h2 style='font:600 16px system-ui'>{_html.escape(stage)}</h2>{findings_html(fs)}" for stage, fs in fnd.items())
    doc = f"<!doctype html><meta charset=utf-8><body style='background:#fcfcfb;color:#0b0b0b;max-width:1200px;margin:24px auto;padding:0 16px'>{head}<h2 style='font:600 16px system-ui'>All findings</h2>{body}<h2 style='font:600 16px system-ui'>Figures</h2>{figs}</body>"
    path.write_text(doc)
    return path
