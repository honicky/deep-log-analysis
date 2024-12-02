import gc
import psutil
import torch


def training_params():
   device = get_device()
   return {
    "BATCH_SIZE": 16 if device.type == "cuda" else 4,
    "MAX_LENGTH": 405,
    "LEARNING_RATE": 1e-4,
    "NUM_EPOCHS": 1
}

def get_device():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    return device

def print_memory_stats(prefix=""):
    """Detailed memory statistics"""

    device = get_device()
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
    elif device.type == "mps":
        allocated = torch.mps.current_allocated_memory() / 1024**3
        reserved = torch.mps.driver_allocated_memory() / 1024**3
    else:
        allocated = reserved = 0

    print(f"\n{prefix} Memory Status:")
    print(f"├── Allocated: {allocated:.2f} GB (actively used by tensors)")
    print(f"├── Reserved:  {reserved:.2f} GB (held by driver)")
    print(f"├── Cached:    {(reserved - allocated):.2f} GB (reserved - allocated)")

    # System memory info
    vm = psutil.virtual_memory()
    print(f"└── System Available: {vm.available / 1024**3:.2f} GB")

def get_gpu_memory_metrics():
    """Get system metrics for logging"""

    device = get_device()
    if device.type == "cuda":
        return {
            "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / (1024**3),
            "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / (1024**3),
        }
    elif device.type == "mps":
        return {
            "gpu_memory_allocated_gb": torch.mps.current_allocated_memory() / (1024**3),
            "gpu_memory_reserved_gb": torch.mps.driver_allocated_memory() / (1024**3),
        }
    return {
        "gpu_memory_allocated_gb": 0,
        "gpu_memory_reserved_gb": 0,
    }

def clear_memory():
    """Explicitly clear memory"""

    device = get_device()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()