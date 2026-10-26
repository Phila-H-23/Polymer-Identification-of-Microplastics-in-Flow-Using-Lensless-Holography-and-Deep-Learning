"""Item 47 speed-up, added 2026-10-13 at Phila's request ("you can make it faster"). No existing script is edited and the
chain is not restarted. Every 15 s this finds the python child processes of the item 47 chain and of its waiter (the
training runs, Part 0 and the analysis) and, if a process is not yet at the target priority class, sets that class and
turns Windows power throttling (EcoQoS) off for it, so the scheduler may use the performance cores. Arguments, seeds,
data and thread counts are unchanged; only run times change. Log: results/logs_item47/boost.log. Stops when the chain
and the waiter have both exited.
usage: python scripts/item47_boost.py <chain pid> <waiter pid> [below|normal]
       python scripts/item47_boost.py once <pid> [below|normal|idle]"""
import os
import sys
import time
import ctypes
from ctypes import wintypes

D = r"C:\Users\haany\OneDrive\research project\dataset 10jan23"
os.chdir(D)
LOG = "results/logs_item47/boost.log"
LEVELS = {"normal": 0x20, "below": 0x4000, "idle": 0x40}

k = ctypes.WinDLL("kernel32", use_last_error=True)


class PE(ctypes.Structure):
    _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                ("th32DefaultHeapID", ctypes.c_void_p), ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
                ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", wintypes.LONG), ("dwFlags", wintypes.DWORD),
                ("szExeFile", ctypes.c_wchar * 260)]


class PTS(ctypes.Structure):
    _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]


k.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
k.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
k.Process32FirstW.argtypes = [ctypes.c_void_p, ctypes.POINTER(PE)]
k.Process32NextW.argtypes = [ctypes.c_void_p, ctypes.POINTER(PE)]
k.CloseHandle.argtypes = [ctypes.c_void_p]
k.OpenProcess.restype = ctypes.c_void_p
k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
k.SetPriorityClass.argtypes = [ctypes.c_void_p, wintypes.DWORD]
k.GetPriorityClass.argtypes = [ctypes.c_void_p]
k.GetPriorityClass.restype = wintypes.DWORD
k.SetProcessInformation.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
k.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD)]


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    open(LOG, "a", encoding="utf-8").write(line + "\n")
    print(line, flush=True)


def alive(pid):
    h = k.OpenProcess(0x1000, False, pid)
    if not h:
        return False
    code = wintypes.DWORD()
    k.GetExitCodeProcess(h, ctypes.byref(code))
    k.CloseHandle(h)
    return code.value == 259


def children(pids):
    """(pid, parent pid) of the python processes started by pids, two levels down."""
    h = k.CreateToolhelp32Snapshot(0x2, 0)
    e = PE()
    e.dwSize = ctypes.sizeof(PE)
    allp, ok = [], k.Process32FirstW(h, ctypes.byref(e))
    while ok:
        allp.append((e.th32ProcessID, e.th32ParentProcessID, e.szExeFile.lower()))
        ok = k.Process32NextW(h, ctypes.byref(e))
    k.CloseHandle(h)
    out, frontier = [], set(pids)
    for _ in range(2):
        nxt = [p for p in allp if p[1] in frontier and p[0] not in pids]
        out += [(p[0], p[1]) for p in nxt if p[2] == "python.exe"]
        frontier = {p[0] for p in nxt}
    return out


def get_class(pid):
    h = k.OpenProcess(0x1000, False, pid)
    if not h:
        return None
    c = k.GetPriorityClass(h)
    k.CloseHandle(h)
    return c


def boost(pid, cls):
    h = k.OpenProcess(0x0200 | 0x1000, False, pid)
    if not h:
        return None, f"OpenProcess failed, error {ctypes.get_last_error()}"
    try:
        s = PTS(1, 0x1, 0x0)  # PROCESS_POWER_THROTTLING_EXECUTION_SPEED in the mask, state off: never EcoQoS
        ok1 = k.SetProcessInformation(h, 4, ctypes.byref(s), ctypes.sizeof(s))
        e1 = ctypes.get_last_error()
        ok2 = k.SetPriorityClass(h, cls)
        e2 = ctypes.get_last_error()
        return k.GetPriorityClass(h), f"throttling off {'ok' if ok1 else f'failed ({e1})'}, class set {'ok' if ok2 else f'failed ({e2})'}"
    finally:
        k.CloseHandle(h)


if __name__ == "__main__":
    if sys.argv[1] == "once":
        pid, cls = int(sys.argv[2]), LEVELS[sys.argv[3] if len(sys.argv) > 3 else "below"]
        got, info = boost(pid, cls)
        log(f"once: pid {pid} priority class now {got:#x} ({info})" if got is not None else f"once: pid {pid}: {info}")
    else:
        chain, waiter = int(sys.argv[1]), int(sys.argv[2])
        lvl = sys.argv[3] if len(sys.argv) > 3 else "below"
        cls = LEVELS[lvl]
        log(f"boost started, pid {os.getpid()}, target {lvl} ({cls:#x}) with power throttling off; chain {chain}, waiter {waiter}")
        while alive(chain) or alive(waiter):
            for pid, ppid in children([chain, waiter]):
                c = get_class(pid)
                if c is not None and c != cls:
                    got, info = boost(pid, cls)
                    log(f"pid {pid} (parent {ppid}): class {c:#x} -> {got:#x} ({info})" if got is not None else f"pid {pid}: {info}")
            time.sleep(15)
        log("chain and waiter have exited; boost stopped")
