"""Item 41 status: results/item41_state.json (machine state, updated by every stage and job) rendered to
results/item41_status.md with a READ THIS FIRST block on top. usage: python scripts/item41_status.py (render only)."""
import os
import json
import time
import hashlib
import datetime as dt

D = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
STATE = os.path.join(D, "results", "item41_state.json")
MD = os.path.join(D, "results", "item41_status.md")
SHA0 = "7221dd9b4a305a7acf0c5043ce2d46dc86138f9385ddc92c80486f87cb7a335a"
ITEMS = [("stage1c", "Stage 1: reconstruct 10 particles, variants A/B/C, autofocus D"),
         ("stage2", "Stage 2: how much of the background is subtracted"),
         ("j1", "J1: single-feature breakdown of the brightness scalars"),
         ("j2", "J2: arm N (--meannorm), 12 runs"),
         ("j3", "J3: CAM share inside the detected box, seed-0 z0 checkpoints")]


def load():
    for _ in range(5):
        try:
            return json.load(open(STATE)) if os.path.exists(STATE) else {}
        except (json.JSONDecodeError, OSError):
            time.sleep(0.5)
    return {}


def update(key, **kw):
    s = load()
    s.setdefault(key, {}).update(kw)
    s.setdefault("_events", []).append(f"{dt.datetime.now():%Y-%m-%d %H:%M} {key}: " + ", ".join(
        f"{k}={v}" for k, v in kw.items() if k in ("status", "note", "reading")))
    tmp = STATE + ".tmp"
    json.dump(s, open(tmp, "w"), indent=1, default=str)
    os.replace(tmp, STATE)
    render()


def render():
    s = load()
    sha = hashlib.sha256(open(os.path.join(D, "train_compact_cnn.py"), "rb").read()).hexdigest()
    done = [n for k, n in ITEMS if s.get(k, {}).get("status") == "done"]
    notdone = [f"{n} ({s.get(k, {}).get('status', 'not started')})" for k, n in ITEMS if s.get(k, {}).get("status") != "done"]
    L = ["# Item 41 status (PET vs PS only)", "", "## READ THIS FIRST", "",
         f"- Updated {dt.datetime.now():%Y-%m-%d %H:%M}.",
         f"- Finished: {'; '.join(done) or 'nothing yet'}.",
         f"- Not finished: {'; '.join(notdone) or 'nothing'}."]
    for k, n in ITEMS:
        if s.get(k, {}).get("reading"):
            L.append(f"- {n}: reading {s[k]['reading']}")
    if "stage1" in s:  # the first Stage 1 run, with a bug (see the deviation)
        L.append(f"- IGNORE the first Stage 1 run (started 10:54, status {s['stage1'].get('status')}): it squared the real part "
                 "of the propagated field instead of its modulus. If it finished, its files results/stage1_particle_NN.png, "
                 "results/stage1_overview.png and results/report_stage1_sharpness.txt are wrong; the corrected run then wrote "
                 "the same names with _corrected (see the Stage 1 section for which names it used).")
    L.append("- Deviations: " + ("; ".join(s.get("_deviations", [])) or "none"))
    L.append(f"- train_compact_cnn.py sha256 at the start {SHA0[:8]}..., now {sha[:8]}... "
             f"({'unchanged' if sha == SHA0 else 'CHANGED'}). results/overnight_release_bgcorr.flag not created.")
    files = sorted(set(f for k, _ in ITEMS for f in s.get(k, {}).get("files", [])))
    L += ["- Files: " + (", ".join(files) or "none yet"), ""]
    for k, n in ITEMS:
        v = s.get(k, {})
        L += [f"## {n}", "", f"- status: {v.get('status', 'not started')}"]
        for kk in ("started", "finished", "note", "reading", "summary", "names"):
            if v.get(kk):
                L.append(f"- {kk}: {v[kk]}")
        L.append("")
    L += ["## Events", ""] + [f"- {e}" for e in s.get("_events", [])[-60:]]
    text = "\n".join(L).replace("\u2014", "-")
    open(MD, "w", encoding="utf-8").write(text + "\n")


def deviation(msg):
    s = load()
    s.setdefault("_deviations", [])
    if msg not in s["_deviations"]:
        s["_deviations"].append(msg)
    tmp = STATE + ".tmp"
    json.dump(s, open(tmp, "w"), indent=1, default=str)
    os.replace(tmp, STATE)
    render()


if __name__ == "__main__":
    render()
