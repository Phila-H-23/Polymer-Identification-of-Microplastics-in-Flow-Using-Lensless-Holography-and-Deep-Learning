"""Item 46 waiter (detached, idle priority): waits until the item 46 chain process exits, then runs
scripts/item46_analysis.py (output to results/logs_item46/analysis.log). usage: python scripts/item46_waiter.py <chain pid>"""
import os
import sys
import time
import ctypes
import subprocess

D = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
os.chdir(D)
sys.path.insert(0, os.path.join(D, "scripts"))
import prio41  # noqa: E402
prio41.set_priority(prio41.IDLE)


def alive(pid):
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.OpenProcess.restype = ctypes.c_void_p
    k.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    k.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    k.CloseHandle.argtypes = [ctypes.c_void_p]
    h = k.OpenProcess(0x1000, 0, pid)
    if not h:
        return False
    code = ctypes.c_uint32()
    k.GetExitCodeProcess(h, ctypes.byref(code))
    k.CloseHandle(h)
    return code.value == 259


pid = int(sys.argv[1])
while alive(pid):
    time.sleep(120)
with open("results/logs_item46/analysis.log", "w", encoding="utf-8") as fh:
    fh.write(time.strftime("%Y-%m-%d %H:%M:%S") + f" chain {pid} exited; running the analysis\n")
    fh.flush()
    rc = subprocess.run([sys.executable, "scripts/item46_analysis.py"], stdout=fh, stderr=subprocess.STDOUT,
                        creationflags=0x40).returncode
    fh.write(time.strftime("%Y-%m-%d %H:%M:%S") + f" analysis exit {rc}\n")
