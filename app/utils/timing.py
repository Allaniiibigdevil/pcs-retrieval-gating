from time import perf_counter


class StageTimer:
    def __init__(self) -> None:
        self._start = perf_counter()
        self._last = self._start
        self.latency_ms: dict[str, float] = {}

    def mark(self, name: str) -> None:
        now = perf_counter()
        self.latency_ms[name] = round((now - self._last) * 1000, 3)
        self._last = now

    def finish(self) -> dict[str, float]:
        self.latency_ms["total"] = round((perf_counter() - self._start) * 1000, 3)
        return self.latency_ms
