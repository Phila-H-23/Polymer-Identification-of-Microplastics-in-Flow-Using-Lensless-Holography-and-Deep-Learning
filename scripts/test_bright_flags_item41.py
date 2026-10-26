"""Item 41 J2: rerun of the unit test scripts/test_bright_flags_exp.py (unchanged) with its output sent to a NEW file,
results/test_bright_flags_item41.txt, and its scratch training outputs to a new scratch folder (unit_bright_item41).
(a) flags off: one fixed-seed epoch with train_compact_cnn.py and with train_compact_cnn_bright.py, best.pt and
results.json identical; (b) --gain-range spread; (c) --meannorm: every transformed patch has mean 1 within 1e-6."""
import os
import sys
import builtins

D = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
sys.path.insert(0, os.path.join(D, "scripts"))
import prio41  # noqa: E402
prio41.set_priority(prio41.IDLE)
import test_bright_flags_exp as t  # noqa: E402

OUT = "results/test_bright_flags_item41.txt"
t.WORK = os.path.join(os.path.dirname(t.WORK), "unit_bright_item41")


def _open(p, *a, **k):
    if p == "results/test_bright_flags_exp.txt":
        if os.path.exists(OUT):
            raise FileExistsError(OUT)
        p = OUT
    return builtins.open(p, *a, **k)


t.open = _open  # main() looks up open() in its module globals first

if __name__ == "__main__":
    t.main()
