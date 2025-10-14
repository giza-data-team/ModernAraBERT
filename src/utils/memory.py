
import os
import psutil
import torch
import logging


def get_memory_usage():
    """
    Get current RAM and VRAM usage statistics.

    Returns:
        Dict containing memory usage information:
            - ram_used_gb: RAM currently in use (GB)
            - ram_percent: Percentage of total RAM in use
            - vram_used_gb: VRAM currently in use (GB)
            - vram_total_gb: Total VRAM available (GB)
            - vram_percent: Percentage of VRAM in use
    """
    memory_stats = {}

    # Get RAM usage
    process = psutil.Process(os.getpid())
    ram_used_bytes = process.memory_info().rss  # Resident Set Size
    ram_used_gb = ram_used_bytes / (1024 ** 3)
    memory_stats["ram_used_gb"] = ram_used_gb
    memory_stats["ram_percent"] = psutil.virtual_memory().percent

    # Get VRAM usage if CUDA is available
    if torch.cuda.is_available():
        try:
            vram_used_bytes = torch.cuda.memory_allocated()
            vram_total_bytes = torch.cuda.get_device_properties(0).total_memory

            vram_used_gb = vram_used_bytes / (1024 ** 3)
            vram_total_gb = vram_total_bytes / (1024 ** 3)
            vram_percent = (vram_used_bytes / vram_total_bytes) * 100

            memory_stats["vram_used_gb"] = vram_used_gb
            memory_stats["vram_total_gb"] = vram_total_gb
            memory_stats["vram_percent"] = vram_percent
        except Exception as e:
            logging.warning(f"Could not get VRAM usage: {e}")
            memory_stats["vram_used_gb"] = 0.0
            memory_stats["vram_total_gb"] = 0.0
            memory_stats["vram_percent"] = 0.0
    else:
        memory_stats["vram_used_gb"] = 0.0
        memory_stats["vram_total_gb"] = 0.0
        memory_stats["vram_percent"] = 0.0

    return memory_stats
