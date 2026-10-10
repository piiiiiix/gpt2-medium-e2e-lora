"""Device-wide NVIDIA telemetry, independently sampled for each run."""
import json
import subprocess
import threading
from datetime import datetime, timezone

class GPUMonitor:
    def __init__(self, path, interval=2):
        self.path, self.interval = path, interval
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    @staticmethod
    def sample():
        stamp = datetime.now(timezone.utc).isoformat()
        try:
            result = subprocess.run(['nvidia-smi',
                '--query-gpu=index,uuid,name,utilization.gpu,temperature.gpu,memory.used,memory.total',
                '--format=csv,noheader,nounits'], capture_output=True, text=True,
                timeout=8, check=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            records = []
            for line in result.stdout.strip().splitlines():
                fields = [x.strip() for x in line.split(',')]
                records.append(dict(time=stamp, gpu=fields[0], uuid=fields[1], name=fields[2],
                    utilization_percent=fields[3], temperature_celsius=fields[4],
                    memory_used_mib=fields[5], memory_total_mib=fields[6], scope='whole_device'))
            return records
        except Exception as exc:
            return [dict(time=stamp, unavailable=True, reason=str(exc))]

    def _loop(self):
        with self.path.open('a', encoding='utf-8', buffering=1) as stream:
            while True:
                for record in self.sample():
                    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                if self.stop_event.wait(self.interval):
                    break

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=12)
