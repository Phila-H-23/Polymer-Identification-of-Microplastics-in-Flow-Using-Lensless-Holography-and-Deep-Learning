"""Item 48, Step 2 checks, printed as PASS/FAIL and kept in results/item48_checks.txt.
(a) --width 64 --extra-convs 0: parameter count 2,365,890 and equal to the depth script's width-64 model.
(b) --width 64 --extra-convs 2: parameter count (stated 5,096,642; a difference is reported, not adjusted).
(c) creates train_compact_cnn_fit.py, a copy of train_compact_cnn_deep.py with --no-aug (training patches built with
    augment=False: no rotation, shift, gain jitter, noise, depth sampling or surround augmentation) and --save-last
    (final-epoch weights also saved as last.pt). A 1-epoch seed-0 run of it with both flags off is compared with the
    same run of the deep script: best.pt tensors and file hash, and results.json apart from args.out (the folders
    differ) and the two new flag entries. Also a 1-epoch run with both flags on: last.pt written and equal to best.pt
    after one epoch, and a different training loss because augmentation is off.
(d) train_compact_cnn.py still begins 7221dd9b; train_compact_cnn_depth.py and train_compact_cnn_deep.py unchanged.
Runs in results/_check_item48 (kept): width 32, --extra-convs 2, every other argument from
results/real_tc_PETvsPS_z0_exp_s0, launched as the chain launches its runs (above-normal priority, throttling off)."""
import os
import sys
import json
import time
import hashlib
import subprocess
import importlib.util

D = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
os.chdir(D)
sys.path.insert(0, os.path.join(D, "scripts"))
import item48_chain as c48  # noqa: E402

CK = "results/_check_item48"
H0 = {"train_compact_cnn.py": "7221dd9b", "train_compact_cnn_depth.py": "dcede2c26e377222",
      "train_compact_cnn_deep.py": "c41dca049f9dd034"}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def mod(path):
    spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def make_fit():
    raw = open("train_compact_cnn_deep.py", "rb").read()
    nl = "\r\n" if b"\r\n" in raw else "\n"
    src = raw.decode("utf-8").replace("\r\n", "\n")
    assert src.startswith('"""')
    hdr = ('"""train_compact_cnn_fit.py (item 48): a copy of train_compact_cnn_deep.py with two new flags, both off by default.\n'
           '  --no-aug     no training augmentation of any kind: the training patches are built with augment=False, so no\n'
           '               rotation, shift, gain jitter, noise, depth sampling or surround augmentation.\n'
           '  --save-last  also saves the final-epoch weights as last.pt (before best.pt is reloaded for the test evaluation).\n'
           'With both flags off the behaviour is that of train_compact_cnn_deep.py, whose description follows.\n\n')
    rep = [('    ap.add_argument("--depth-aug", choices=["none", "all21", "shifted20"], default="none",\n',
            '    ap.add_argument("--no-aug", action="store_true",\n'
            '                    help="item 48: no training augmentation of any kind (rotation, shift, gain, noise, depth, surround)")\n'
            '    ap.add_argument("--save-last", action="store_true", help="item 48: also save the final-epoch weights as last.pt")\n'
            '    ap.add_argument("--depth-aug", choices=["none", "all21", "shifted20"], default="none",\n'),
           ('        train_ds = mk("train", True)\n',
            '        train_ds = mk("train", not args.no_aug)  # --no-aug: augment=False for the training patches\n'),
           ('                                augment=True, mean=mean, std=std, patch_norm=args.patch_norm,\n',
            '                                augment=not args.no_aug, mean=mean, std=std, patch_norm=args.patch_norm,\n'),
           ('    # Final evaluation with the best-on-validation checkpoint\n',
            '    if args.save_last:  # final-epoch weights, saved before best.pt is reloaded below\n'
            '        torch.save(model.state_dict(), os.path.join(args.out, "last.pt"))\n\n'
            '    # Final evaluation with the best-on-validation checkpoint\n')]
    for a, b in rep:
        assert src.count(a) == 1, a
        src = src.replace(a, b)
    assert not os.path.exists("train_compact_cnn_fit.py")
    open("train_compact_cnn_fit.py", "w", encoding="utf-8", newline=nl).write(hdr + src[3:])


def run1(script, name, flags=()):
    out = f"{CK}/{name}"
    assert not os.path.exists(out)
    with open(f"{CK}/{name}.log", "w", encoding="utf-8") as fh:
        p, cls, thr = c48.launch(c48.cmd(script, "n", "original", 0, out, 32, 2, epochs=1, flags=flags),
                                 stdout=fh, stderr=subprocess.STDOUT)
        rc = p.wait()
    assert rc == 0, (name, rc)
    return out


def main():
    import torch
    L, ok = [], {}
    before = {k: sha(k) for k in H0}
    deep, depth = mod("train_compact_cnn_deep.py"), mod("train_compact_cnn_depth.py")
    n = lambda m, w, **kw: sum(p.numel() for p in m.CompactHoloCNN(1, 2, w, 0.3, **kw).parameters())
    a64, d64 = n(deep, 64, extra_convs=0), n(depth, 64)
    ok["a"] = a64 == 2365890 and a64 == d64
    L.append(f"(a) width 64, --extra-convs 0: {a64:,} parameters; depth script width 64: {d64:,} -> {'PASS' if ok['a'] else 'FAIL'}")
    b64 = n(deep, 64, extra_convs=2)
    L.append(f"(b) width 64, --extra-convs 2: {b64:,} parameters (stated 5,096,642; difference {b64 - 5096642:+,})")
    os.makedirs(CK, exist_ok=True)
    make_fit()
    t0 = time.time()
    rd, rf = run1("train_compact_cnn_deep.py", "deep_e1"), run1("train_compact_cnn_fit.py", "fit_e1")
    rg = run1("train_compact_cnn_fit.py", "fit_e1_flags", flags=("--no-aug", "--save-last"))
    jd, jf, jg = (json.load(open(f"{r}/results.json")) for r in (rd, rf, rg))
    for j in (jd, jf):
        j["args"].pop("out")
    new = {k: jf["args"].pop(k) for k in ("no_aug", "save_last")}
    same_json = jd == jf
    sd, sf = (torch.load(f"{r}/best.pt", map_location="cpu") for r in (rd, rf))
    same_t = sd.keys() == sf.keys() and all(torch.equal(sd[k], sf[k]) for k in sd)
    same_file = sha(f"{rd}/best.pt") == sha(f"{rf}/best.pt")
    sg, sl = (torch.load(f"{rg}/{f}", map_location="cpu") for f in ("best.pt", "last.pt"))
    last_eq = all(torch.equal(sg[k], sl[k]) for k in sg)
    loss_d, loss_g = jf["history"][0]["train_loss"], jg["history"][0]["train_loss"]
    flags_on = jg["args"]["no_aug"] is True and jg["args"]["save_last"] is True and last_eq and loss_g != loss_d
    ok["c"] = same_json and same_t and same_file and new == {"no_aug": False, "save_last": False} and flags_on
    L.append(f"(c) train_compact_cnn_fit.py created (sha256 {sha('train_compact_cnn_fit.py')[:16]}). 1-epoch seed-0 runs "
             f"({time.time() - t0:.0f} s): flags off vs deep script: results.json identical apart from args.out and the new "
             f"entries {new}: {same_json}; best.pt tensors identical: {same_t}; best.pt file hash identical: {same_file}. "
             f"Flags on: last.pt written and equal to best.pt after 1 epoch: {last_eq}; epoch-1 training loss {loss_g:.4f} "
             f"against {loss_d:.4f} with augmentation -> {'PASS' if ok['c'] else 'FAIL'}")
    after = {k: sha(k) for k in H0}
    ok["d"] = before == after and all(after[k].startswith(v) for k, v in H0.items())
    L.append("(d) " + ", ".join(f"{k} {after[k][:16]}" for k in H0) + f" (unchanged: {before == after}) -> {'PASS' if ok['d'] else 'FAIL'}")
    L.append(f"Written {time.strftime('%Y-%m-%d %H:%M')}.")
    text = "\n".join(L)
    assert "\u2014" not in text and ("signif" + "icant") not in text.lower() and ("valid" + "ated") not in text.lower()
    open("results/item48_checks.txt", "x", encoding="utf-8").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
