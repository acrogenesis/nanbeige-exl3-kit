"""Memory/cache observations shared by the occupied-context benchmark."""
from bench_exl3 import CacheLayer_quant, torch

def cache_kwargs(bits):
    return {"layer_type": CacheLayer_quant, "k_bits": bits, "v_bits": bits} if bits else {}

def cache_bytes(cache):
    return 0 if cache is None else sum(int(layer.storage_size()) for layer in cache.layers.values())

def memory():
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return {"free_bytes": free, "total_bytes": total,
            "torch_allocated_bytes": torch.cuda.memory_allocated(),
            "torch_reserved_bytes": torch.cuda.memory_reserved(),
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved()}
