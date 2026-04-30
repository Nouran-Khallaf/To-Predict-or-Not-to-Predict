"""GPU-safe timing helpers."""

import time
import torch


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def now():
    sync_cuda()
    return time.perf_counter()


class Timer:
    def __init__(self):
        self.t0 = None

    def start(self):
        self.t0 = now()

    def stop(self):
        return now() - self.t0
