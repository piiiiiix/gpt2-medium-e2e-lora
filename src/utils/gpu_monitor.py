"""GPU 监控模块：独立后台采样设备占用率、核心温度和显存，写入对应任务日志。"""

import json
import subprocess
import threading
import xml.etree.ElementTree as ET
from datetime import datetime, timezone


class GPUMonitor:
    """每次运行独立的 GPU 采样器，用后台线程记录整张设备的资源状态。"""

    def __init__(self, path, interval=2):
        """设置输出路径和采样周期，初始化停止事件与后台线程。"""
        self.path, self.interval = path, interval
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    @staticmethod
    def sample():
        """调用 nvidia-smi 读取各 GPU 指标；不可用时返回带原因的记录。"""
        stamp = datetime.now(timezone.utc).isoformat()
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,name,utilization.gpu,temperature.gpu,memory.used,memory.total,power.draw,power.limit,clocks.current.graphics,clocks.current.sm,clocks.current.memory,pstate,clocks_event_reasons.active",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=8,
                check=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            records = []
            for line in result.stdout.strip().splitlines():
                fields = [x.strip() for x in line.split(",")]
                records.append(
                    dict(
                        time=stamp,
                        gpu=fields[0],
                        uuid=fields[1],
                        name=fields[2],
                        utilization_percent=fields[3],
                        temperature_celsius=fields[4],
                        memory_used_mib=fields[5],
                        memory_total_mib=fields[6],
                        scope="whole_device",
                        **dict(
                            zip(
                                (
                                    "power_draw_watts",
                                    "power_limit_watts",
                                    "core_clock_mhz",
                                    "sm_clock_mhz",
                                    "memory_clock_mhz",
                                    "performance_state",
                                    "clock_event_reasons_active",
                                ),
                                (
                                    None if value in ("N/A", "[N/A]", "[Not Supported]") else value
                                    for value in fields[7:]
                                ),
                            )
                        ),
                    )
                )
            # 电压没有通用 CSV 查询字段；单独查询，失败不丢弃其他采样。
            try:
                voltage = subprocess.run(
                    ["nvidia-smi", "-q", "-x", "-d", "VOLTAGE"],
                    capture_output=True,
                    text=True,
                    timeout=8,
                    check=True,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                devices = ET.fromstring(voltage.stdout).findall("gpu")
                for record, device in zip(records, devices):
                    value = device.findtext("voltage/graphics_volt")
                    record["core_voltage_mv"] = (
                        float(value.split()[0]) if value and "N/A" not in value else None
                    )
            except Exception as exc:
                for record in records:
                    record["voltage_unavailable_reason"] = str(exc)
            for record in records:
                if record.get("core_voltage_mv") is None:
                    record["core_voltage_mv"] = None
                    record.setdefault(
                        "voltage_unavailable_reason", "Driver did not report graphics voltage"
                    )
            return records
        except Exception as exc:
            return [dict(time=stamp, unavailable=True, reason=str(exc))]

    def _loop(self):
        """周期采样并逐行写入 JSONL，直到收到停止事件。"""
        from utils.windows_gpu_memory import WindowsGPUMemory

        memory = WindowsGPUMemory()
        try:
            with self.path.open("a", encoding="utf-8", buffering=1) as stream:
                while True:
                    records = self.sample()
                    # 独立 LUID 命名空间，每轮只保存一次，不重复归属给不同 NVIDIA GPU。
                    records[0]["wddm_memory"] = memory.sample()
                    for record in records:
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    if self.stop_event.wait(self.interval):
                        break
        finally:
            memory.close()

    def start(self):
        """启动本次任务的后台 GPU 采样线程。"""
        self.thread.start()

    def stop(self):
        """通知采样结束并等待线程刷新日志、释放文件。"""
        self.stop_event.set()
        self.thread.join(timeout=12)
