import time


class Timer:
    def __init__(self) -> None:
        self._started = False
        self._start_time = None
        self._elapsed = -1

    def start(self):
        if not self._started:
            self._started = True
            self._elapsed = -1
            self._start_time = time.time()

    def stop(self):
        if self._started:
            self._started = False
            end_time = time.time()
            self._elapsed = end_time - self._start_time

    @property
    def elapsed(self):
        return self._elapsed
