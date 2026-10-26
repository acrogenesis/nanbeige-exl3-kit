import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("EXL3_GRAPHS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import torch
from exllamav3 import Config, Model, Cache, CacheLayer_quant, Tokenizer, Generator, Job, TopPSampler
from transformers import PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / "2026-10-25-100tps"
TARGET = Path.home() / ".cache/huggingface/hub/models--ProCreations--Nanbeige4.3-33B-EXL3/snapshots/56a23706262458b23ba7dde5ba1eea053a109544"
DRAFT = Path(json.loads((PREVIOUS / "models.json").read_text())["draft"]["path"])

def cache_kwargs(bits):
    return {"layer_type": CacheLayer_quant, "k_bits": bits, "v_bits": bits} if bits else {}

def cache_bytes(cache):
    if cache is None:
        return 0
    return sum(int(layer.storage_size()) for layer in cache.layers.values())

def memory():
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return {"free_bytes": free, "total_bytes": total,
            "torch_allocated_bytes": torch.cuda.memory_allocated(),
            "torch_reserved_bytes": torch.cuda.memory_reserved(),
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved()}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capacity", type=int, default=8192)
    p.add_argument("--target-bits", type=int, default=0)
    p.add_argument("--draft-bits", type=int, default=0)
    p.add_argument("--label", required=True)
    p.add_argument("--geometry-only", action="store_true")
    p.add_argument("--no-draft", action="store_true")
    a = p.parse_args()
    report = {"settings": vars(a), "state": "initializing", "started_epoch": time.time()}
    out = ROOT / (a.label + ".json")
    def save():
        out.write_text(json.dumps(report, indent=2) + "\n")
    save()
    config = Config.from_directory(str(TARGET))
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=a.capacity, **cache_kwargs(a.target_bits))
    dc = draft = draft_cache = None
    if not a.no_draft:
        dc = Config.from_directory(str(DRAFT))
        draft = Model.from_config(dc)
        draft_cache = Cache(draft, max_num_tokens=a.capacity, **cache_kwargs(a.draft_bits))
    report["target_cache_bytes"] = cache_bytes(cache)
    report["draft_cache_bytes"] = cache_bytes(draft_cache)
    report["cache_bytes_per_token"] = (cache_bytes(cache) + cache_bytes(draft_cache)) / a.capacity
    report["model_max_position_embeddings"] = config.max_position_embeddings
    if a.geometry_only:
        report["state"] = "geometry_only"
        save()
        print(json.dumps(report), flush=True)
        return
    try:
        report["memory_before_load"] = memory()
        report["state"] = "loading_target"
        save()
        model.load(device=torch.device("cuda:0"), max_batch_size=1,
                   max_chunk_size=2048, max_output_size=2048)
        report["memory_after_target"] = memory()
        report["state"] = "loading_draft"
        save()
        if draft is not None:
            draft.load(device=torch.device("cuda:0"), max_batch_size=1,
                       max_chunk_size=2048, max_output_size=2048)
            draft.attach_to(model)
        report["memory_after_models"] = memory()
        tokenizer = Tokenizer.from_config(config)
        hf = PreTrainedTokenizerFast.from_pretrained(str(TARGET), local_files_only=True)
        generator = Generator(model=model, cache=cache, tokenizer=tokenizer,
            draft_model=draft, draft_cache=draft_cache, num_draft_tokens=7,
            dynamic_draft_tokens=False, max_batch_size=1, max_chunk_size=2048,
            record_draft_stats=True, sparse_draft_head=16384 if draft is not None else 0)
        for index in range(2):
            ids = hf.apply_chat_template([{"role": "user", "content":
                "Explain the architecture and recovery behavior of a database write-ahead log in detail."}],
                tokenize=True, add_generation_prompt=True, enable_thinking=False, preserve_thinking=False)
            job = Job(input_ids=torch.tensor([ids], dtype=torch.long), max_new_tokens=128,
                sampler=TopPSampler(.95, temperature=.6), seed=20261026,
                stop_conditions=[166101], token_healing=False, stop_on_loop=None)
            generator.enqueue(job)
            terminal = None
            while generator.num_remaining_jobs():
                for result in generator.iterate():
                    if result.get("eos"):
                        terminal = {k: v for k, v in result.items() if isinstance(v, (str, int, float, bool, type(None)))}
            assert terminal is not None and terminal["new_tokens"] >= 64, terminal
            report[f"smoke_{index}"] = terminal
        report["memory_after_smoke"] = memory()
        report["state"] = "passed"
        report["completed_epoch"] = time.time()
        save()
        print(json.dumps(report), flush=True)
    except Exception as error:
        report["state"] = "failed"
        report["error"] = repr(error)
        report["completed_epoch"] = time.time()
        save()
        raise
    finally:
        if draft is not None:
            draft.unload()
            dc.stc.close()
        model.unload()
        config.stc.close()

if __name__ == "__main__":
    main()
