"""Item 41 stage 3 status: results/item41_stage3_state.json (machine state) rendered to
results/item41_stage3_status.md with a READ THIS FIRST block on top (summary and readings table once known, what
finished, what did not, deviations, file list). usage: python scripts/item41_stage3_status.py (render only)."""
import os
import json
import time
import hashlib
import datetime as dt

D = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
STATE = os.path.join(D, "results", "item41_stage3_state.json")
MD = os.path.join(D, "results", "item41_stage3_status.md")
SHA0 = "7221dd9b4a305a7acf0c5043ce2d46dc86138f9385ddc92c80486f87cb7a335a"
ITEMS = [("estimate", "Time estimate (before building)"),
         ("build", "Step 1: build the V3 recon_s0 dataset (561 real + 561 particle-free patches)"),
         ("checks", "Step 1 checks (a) to (d)"),
         ("train", "Step 2: 16 training runs"),
         ("report", "Step 2: evaluation and report")]


def load():
    for _ in range(5):
        try:
            return json.load(open(STATE)) if os.path.exists(STATE) else {}
        except (json.JSONDecodeError, OSError):
            time.sleep(0.5)
    return {}


def _save(s):
    tmp = STATE + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(s, fh, indent=1, default=str)
    for i in range(30):  # a reader (or OneDrive) holding the file blocks the replace on Windows: retry
        try:
            os.replace(tmp, STATE)
            return
        except PermissionError:
            time.sleep(1)
    os.replace(tmp, STATE)


def update(key, **kw):
    s = load()
    s.setdefault(key, {}).update(kw)
    s.setdefault("_events", []).append(f"{dt.datetime.now():%Y-%m-%d %H:%M} {key}: " + ", ".join(
        f"{k}={v}" for k, v in kw.items() if k in ("status", "note", "reading")))
    _save(s)
    render()


def deviation(msg):
    s = load()
    s.setdefault("_deviations", [])
    if msg not in s["_deviations"]:
        s["_deviations"].append(msg)
    _save(s)
    render()


def top(summary_lines, table_lines):
    s = load()
    s["_top"] = dict(summary=summary_lines, table=table_lines)
    _save(s)
    render()


def render():
    s = load()
    sha = hashlib.sha256(open(os.path.join(D, "train_compact_cnn.py"), "rb").read()).hexdigest()
    done = [n for k, n in ITEMS if s.get(k, {}).get("status") == "done"]
    notdone = [f"{n} ({s.get(k, {}).get('status', 'not started')})" for k, n in ITEMS if s.get(k, {}).get("status") != "done"]
    L = ["# Item 41, stage 3: V3 rebuild of recon_s0, PET vs PS, old 561-patch set", "", "## READ THIS FIRST", ""]
    if s.get("_top"):
        L += ["Summary:"] + [f"{i + 1}. {x}" for i, x in enumerate(s["_top"]["summary"])] + ["", "Readings:", ""]
        L += s["_top"]["table"] + [""]
    L += [f"- Updated {dt.datetime.now():%Y-%m-%d %H:%M}.",
          f"- Finished: {'; '.join(done) or 'nothing yet'}.",
          f"- Not finished: {'; '.join(notdone) or 'nothing'}."]
    for k, n in ITEMS:
        if s.get(k, {}).get("reading"):
            L.append(f"- {n}: {s[k]['reading']}")
    L.append("- Deviations: " + ("; ".join(s.get("_deviations", [])) or "none"))
    L.append(f"- train_compact_cnn.py sha256 at the start {SHA0[:8]}..., now {sha[:8]}... "
             f"({'unchanged' if sha == SHA0 else 'CHANGED'}). results/overnight_release_bgcorr.flag not created.")
    files = sorted(set(f for k, _ in ITEMS for f in s.get(k, {}).get("files", [])))
    L += ["- Files: " + (", ".join(files) or "none yet"), ""]
    for k, n in ITEMS:
        v = s.get(k, {})
        L += [f"## {n}", "", f"- status: {v.get('status', 'not started')}"]
        for kk in ("started", "finished", "note", "reading", "summary", "progress"):
            if v.get(kk):
                L.append(f"- {kk}: {v[kk]}")
        L.append("")
    L += ["## Events", ""] + [f"- {e}" for e in s.get("_events", [])[-80:]]
    text = "\n".join(L).replace("\u2014", "-")
    open(MD, "w", encoding="utf-8").write(text + "\n")


if __name__ == "__main__":
    render()
