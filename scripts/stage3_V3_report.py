"""Item 41 stage 3, Step 2 report (run by scripts/stage3_V3_chain.py step2 after the 16 runs; idle priority):
evaluation of the saved runs (scripts/eval_track_ci_exp.py analyse()), readings T, R, M, the unclipped-only table and
the 7-flagged-dropped table -> results/report_recon_V3_PETvsPS.txt, and the summary and readings table at the top of
results/item41_stage3_status.md. New files only."""
import os
import sys
import json
import time

D = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
os.chdir(D)
sys.path.insert(0, os.path.join(D, "scripts"))
import stage3_V3_chain as ch  # noqa: E402  (sets idle priority)
import item41_stage3_status as st  # noqa: E402


def main():
    rj = json.load(open(ch.CFG["runs_json"]))
    runs = {d: {k: {int(s): p for s, p in v.items()} for k, v in kinds.items()} for d, kinds in rj["runs"].items()}
    checks = json.load(open(ch.CFG["checks_json"]))
    R, M, summ, summary, table = ch.report(runs, rj["failed"], rj["timing"], checks)
    st.top(summary, table)
    st.update("report", status="done", finished=time.strftime("%Y-%m-%d %H:%M"),
              reading=f"{R}, {M}; T original {summ['original']['T']}, reverse {summ['reverse']['T']}",
              files=[ch.CFG["report"], ch.CFG["pred_csv"], ch.CFG["trackci_json"], "scripts/stage3_V3_report.py",
                     "scripts/unclipped_tracks.py"])
    ch.log(f"report written: {R}, {M}")


if __name__ == "__main__":
    main()
