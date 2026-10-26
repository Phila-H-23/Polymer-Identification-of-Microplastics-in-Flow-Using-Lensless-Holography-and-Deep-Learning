"""Process priority for item 41 that actually takes effect. The earlier pattern
ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40) passes the pseudo-handle
as a 32-bit int and fails with ERROR_INVALID_HANDLE (6) on 64-bit Python, so it never changed anything. Here the
handle type is declared and the result is read back."""
import ctypes

_k = ctypes.WinDLL("kernel32", use_last_error=True)
_k.GetCurrentProcess.restype = ctypes.c_void_p
_k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
_k.SetPriorityClass.restype = ctypes.c_int
_k.GetPriorityClass.argtypes = [ctypes.c_void_p]
_k.GetPriorityClass.restype = ctypes.c_uint32
IDLE, NORMAL = 0x40, 0x20


def set_priority(cls):
    h = _k.GetCurrentProcess()
    if not _k.SetPriorityClass(h, cls):
        raise OSError(f"SetPriorityClass failed, error {ctypes.get_last_error()}")
    got = _k.GetPriorityClass(h)
    if got != cls:
        raise OSError(f"priority class is {got:#x}, wanted {cls:#x}")
    return got
