"""Item 47 unit tests of train_compact_cnn_deep.py against train_compact_cnn_depth.py (model only, no training)."""
import os
import sys
import hashlib
import importlib.util
import torch

D = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
os.chdir(D)


def mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


dep, deep = mod("train_compact_cnn_depth.py", "dep"), mod("train_compact_cnn_deep.py", "deep")
npar = lambda n: sum(p.numel() for p in n.parameters() if p.requires_grad)
ok_a = True
for w, expect in ((32, 593122), (64, 2365890)):
    a, b = dep.CompactHoloCNN(1, 2, w, 0.3), deep.CompactHoloCNN(1, 2, w, 0.3, extra_convs=0)
    sa, sb = a.state_dict(), b.state_dict()
    same = list(sa) == list(sb) and all(sa[k].shape == sb[k].shape for k in sa) and npar(a) == npar(b) == expect
    b.load_state_dict(sa)
    a.eval(); b.eval()
    x = torch.randn(4, 1, 64, 64)
    with torch.no_grad():
        same = same and torch.equal(a(x), b(x))
    ok_a = ok_a and same
    print(f"(a) width {w}: K=0 keys/shapes/count identical, count {npar(b):,} (expected {expect:,}), same outputs with the same weights -> {'PASS' if same else 'FAIL'}")
k2 = deep.CompactHoloCNN(1, 2, 32, 0.3, extra_convs=2)
k2.eval()
with torch.no_grad():
    out = k2(torch.randn(4, 1, 64, 64))
convs = sum(isinstance(m_, torch.nn.Conv2d) for m_ in k2.modules())
lins = sum(isinstance(m_, torch.nn.Linear) for m_ in k2.modules())
ok_b = tuple(out.shape) == (4, 2) and convs == 17 and lins == 1
print(f"(b) K=2 width 32: output {tuple(out.shape)}, {convs} conv + {lins} linear, parameters {npar(k2):,} (stated 1,276,514; difference "
      f"{npar(k2) - 1276514:+,}) -> {'PASS' if ok_b else 'FAIL'}")
h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
ok_c = h("train_compact_cnn.py").startswith("7221dd9b") and h("train_compact_cnn_depth.py").startswith("dcede2c26e377222")
print(f"(c) train_compact_cnn.py {h('train_compact_cnn.py')[:8]}, train_compact_cnn_depth.py {h('train_compact_cnn_depth.py')[:16]} "
      f"(unchanged), train_compact_cnn_deep.py {h('train_compact_cnn_deep.py')[:16]} -> {'PASS' if ok_c else 'FAIL'}")
