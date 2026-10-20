"""使用 Windows PDH 读取 WDDM 专用/共享 GPU 内存，不启动辅助进程。"""

import ctypes as ct
import os
import re
from ctypes import wintypes as wt


class CounterValue(ct.Structure):
    """PDH_FMT_DOUBLE 的状态与 8 字节 union 存储。"""

    _fields_ = [("status", wt.DWORD), ("value", ct.c_double)]


class CounterItem(ct.Structure):
    """PDH 返回的实例名及格式化值。"""

    _fields_ = [("name", wt.LPWSTR), ("value", CounterValue)]


class WindowsGPUMemory:
    """每个采样线程持有一个 PDH query，以 LUID 标识适配器。"""

    def __init__(self, pid=None):
        self.pid = os.getpid() if pid is None else pid
        self.query = wt.HANDLE()
        self.counters = {}
        self.error = None
        if os.name != "nt":
            self.error = "WDDM counters require Windows"
            return
        try:
            self.api = ct.WinDLL("pdh")
            signatures = {
                "PdhOpenQueryW": [wt.LPCWSTR, ct.c_size_t, ct.POINTER(wt.HANDLE)],
                "PdhAddEnglishCounterW": [
                    wt.HANDLE,
                    wt.LPCWSTR,
                    ct.c_size_t,
                    ct.POINTER(wt.HANDLE),
                ],
                "PdhCollectQueryData": [wt.HANDLE],
                "PdhGetFormattedCounterArrayW": [
                    wt.HANDLE,
                    wt.DWORD,
                    ct.POINTER(wt.DWORD),
                    ct.POINTER(wt.DWORD),
                    ct.c_void_p,
                ],
                "PdhCloseQuery": [wt.HANDLE],
            }
            for name, args in signatures.items():
                fn = getattr(self.api, name)
                fn.argtypes, fn.restype = args, wt.DWORD
            self._check(self.api.PdhOpenQueryW(None, 0, ct.byref(self.query)))
            for scope, category in [
                ("adapter", "GPU Adapter Memory"),
                ("process", "GPU Process Memory"),
            ]:
                for metric in ("Dedicated", "Shared"):
                    handle = wt.HANDLE()
                    self._check(
                        self.api.PdhAddEnglishCounterW(
                            self.query, f"\\{category}(*)\\{metric} Usage", 0, ct.byref(handle)
                        )
                    )
                    self.counters[scope, metric.lower()] = handle
        except Exception as exc:
            self.error = str(exc)
            self.close()

    @staticmethod
    def _check(status):
        """PDH 状态码以十六进制保留，便于定位权限或驱动问题。"""
        if status:
            raise OSError(f"PDH error 0x{status:08x}")

    def _values(self, counter):
        """获取动态实例数组，实例数变化时有限重试。"""
        for _ in range(3):
            size, count = wt.DWORD(), wt.DWORD()
            status = self.api.PdhGetFormattedCounterArrayW(
                counter, 0x200, ct.byref(size), ct.byref(count), None
            )
            if status == 0 and size.value == 0:
                return {}
            if status != 0x800007D2:  # PDH_MORE_DATA
                self._check(status)
            buffer = ct.create_string_buffer(size.value)
            status = self.api.PdhGetFormattedCounterArrayW(
                counter, 0x200, ct.byref(size), ct.byref(count), buffer
            )
            if status == 0x800007D2:
                continue
            self._check(status)
            items = ct.cast(buffer, ct.POINTER(CounterItem))
            return {
                items[i].name: items[i].value.value / (1024 * 1024)
                for i in range(count.value)
                if items[i].value.status in (0, 1)
            }
        raise OSError("WDDM counter instances changed repeatedly")

    def sample(self):
        """返回独立 WDDM 记录；不把 LUID 猜测映射到 NVIDIA GPU index。"""
        if self.error:
            return {"pid": self.pid, "unavailable_reason": self.error, "adapters": []}
        try:
            self._check(self.api.PdhCollectQueryData(self.query))
            values = {key: self._values(handle) for key, handle in self.counters.items()}
            adapters = {}
            for (scope, metric), instances in values.items():
                for name, value in instances.items():
                    if scope == "process":
                        match = re.fullmatch(r"pid_(\d+)_(luid_.+)", name)
                        if not match or int(match[1]) != self.pid:
                            continue
                        name = match[2]
                    row = adapters.setdefault(name, {"adapter": name})
                    row[f"{scope}_{metric}_mib"] = value
            if not adapters:
                raise OSError("No WDDM memory counter instances")
            return {"pid": self.pid, "adapters": list(adapters.values())}
        except Exception as exc:
            return {"pid": self.pid, "unavailable_reason": str(exc), "adapters": []}

    def close(self):
        """关闭 query；可在初始化失败或线程退出时重复调用。"""
        if self.query:
            self.api.PdhCloseQuery(self.query)
            self.query = wt.HANDLE()
