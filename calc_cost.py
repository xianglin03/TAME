# <code>
import os
import time
import psutil
import tracemalloc
from contextlib import contextmanager

try:
    import torch
except ImportError:
    torch = None

_HAS_NVML = False

def _get_gpu_stats(device_index=0):
    stats = {
        "gpu_util_percent": None,
        "gpu_mem_used_mb": None,
    }
    if not _HAS_NVML:
        return stats
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        stats["gpu_util_percent"] = float(util.gpu)
        stats["gpu_mem_used_mb"] = mem.used / 1024**2
    except Exception:
        pass
    return stats


@contextmanager
def profile_block(name="block", device=None, warmup_cuda=True):
    """
    Usage:
        with profile_block("sampling_step"):
            # <test>
            ...
            # </test>
    """
    process = psutil.Process(os.getpid())

    use_cuda = (
        torch is not None and
        torch.cuda.is_available() and
        (device is None or "cuda" in str(device))
    )

    if use_cuda:
        if device is None:
            device = torch.device("cuda:0")
        else:
            device = torch.device(device)

        if warmup_cuda:
            torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    # CPU / RAM start
    cpu_times_start = process.cpu_times()
    mem_start = process.memory_info().rss / 1024**2  # MB
    cpu_percent_start = psutil.cpu_percent(interval=None)

    # Python allocation tracing
    tracemalloc.start()

    # Optional GPU stats start
    gpu_start = _get_gpu_stats(0)

    t0 = time.perf_counter()

    try:
        yield
    finally:
        if use_cuda:
            torch.cuda.synchronize(device)

        t1 = time.perf_counter()
        current_py_mem, peak_py_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        cpu_times_end = process.cpu_times()
        mem_end = process.memory_info().rss / 1024**2  # MB
        cpu_percent_end = psutil.cpu_percent(interval=None)

        gpu_end = _get_gpu_stats(0)

        wall_time = t1 - t0
        cpu_time = (
            (cpu_times_end.user - cpu_times_start.user) +
            (cpu_times_end.system - cpu_times_start.system)
        )

        avg_cpu_util = None
        if wall_time > 0:
            # rough per-process CPU utilization
            avg_cpu_util = 100.0 * cpu_time / wall_time

        result = {
            "name": name,
            "wall_time_sec": wall_time,
            "cpu_time_sec": cpu_time,
            "avg_cpu_util_percent": avg_cpu_util,
            "rss_mem_start_mb": mem_start,
            "rss_mem_end_mb": mem_end,
            "rss_mem_delta_mb": mem_end - mem_start,
            "peak_python_alloc_mb": peak_py_mem / 1024**2,
        }

        if use_cuda:
            result["peak_gpu_mem_alloc_mb"] = (
                torch.cuda.max_memory_allocated(device) / 1024**2
            )
            result["peak_gpu_mem_reserved_mb"] = (
                torch.cuda.max_memory_reserved(device) / 1024**2
            )

        if gpu_end["gpu_util_percent"] is not None:
            result["gpu_util_start_percent"] = gpu_start["gpu_util_percent"]
            result["gpu_util_end_percent"] = gpu_end["gpu_util_percent"]
            result["gpu_mem_used_start_mb"] = gpu_start["gpu_mem_used_mb"]
            result["gpu_mem_used_end_mb"] = gpu_end["gpu_mem_used_mb"]

        print("\n" + "=" * 60)
        print(f"[PROFILE] {name}")
        for k, v in result.items():
            if k == "name":
                continue
            if isinstance(v, float):
                print(f"{k:>28s}: {v:.4f}")
            else:
                print(f"{k:>28s}: {v}")
        print("=" * 60 + "\n")


# -----------------------------
# Example
# -----------------------------
if __name__ == "__main__":
    with profile_block("TAME_sampling", device="cuda:0"):
        # <test>
        # Put your TAME sampling code here
        # Example:
        # samples = sample_with_alignment(model, predictor, cond, num_steps=1000)
        # </test>
        pass