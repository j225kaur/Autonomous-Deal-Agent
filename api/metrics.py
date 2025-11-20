import time
from typing import Dict, Any
from collections import defaultdict
import threading

class MetricsCollector:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MetricsCollector, cls).__new__(cls)
                cls._instance.reset()
        return cls._instance

    def reset(self):
        self.counters = defaultdict(int)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(list)

    def inc(self, name: str, value: int = 1, labels: Dict[str, str] = None):
        key = self._format_key(name, labels)
        self.counters[key] += value

    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        key = self._format_key(name, labels)
        self.gauges[key] = value

    def observe(self, name: str, value: float, labels: Dict[str, str] = None):
        key = self._format_key(name, labels)
        self.histograms[key].append(value)

    def _format_key(self, name: str, labels: Dict[str, str] = None) -> str:
        if not labels:
            return name
        label_str = ",".join([f'{k}="{v}"' for k, v in sorted(labels.items())])
        return f'{name}{{{label_str}}}'

    def generate_prometheus_output(self) -> str:
        lines = []
        
        # Counters
        for key, val in self.counters.items():
            lines.append(f"# TYPE {key.split('{')[0]} counter")
            lines.append(f"{key} {val}")

        # Gauges
        for key, val in self.gauges.items():
            lines.append(f"# TYPE {key.split('{')[0]} gauge")
            lines.append(f"{key} {val}")

        # Histograms (simplified: just sum and count for now)
        for key, vals in self.histograms.items():
            base_name = key.split('{')[0]
            label_part = key[len(base_name):] if '{' in key else ''
            
            lines.append(f"# TYPE {base_name} histogram")
            lines.append(f"{base_name}_sum{label_part} {sum(vals)}")
            lines.append(f"{base_name}_count{label_part} {len(vals)}")

        return "\n".join(lines)

metrics = MetricsCollector()
