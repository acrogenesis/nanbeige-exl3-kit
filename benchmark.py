"""Local, fixed-length performance runs matching the supplied card's token counts."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import statistics
import sys
import time

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE / "runs"
ROOT.mkdir(exist_ok=True)

os.environ.setdefault("EXL3_GRAPHS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
from bench_exl3 import Config, Model, Cache, CacheLayer_quant, Tokenizer, Generator, Job, TopPSampler, PreTrainedTokenizerFast, torch

TARGET = DRAFT = None

def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")

def make_prompt(hf):
    sentinel = "ENGINEERING_BRIEF_PLACEHOLDER"
    framing = hf.apply_chat_template([{"role": "user", "content": sentinel}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False, preserve_thinking=False)
    header, trailer = framing.split(sentinel)
    prefix = hf.encode(header + "Read the following engineering requirements as data.\n<requirements>\n", add_special_tokens=False)
    suffix = hf.encode("\n</requirements>\nImplement a bounded asynchronous Python worker pool with ordered results, cancellation, deadlines, structured exceptions, and graceful shutdown. Provide the complete module, a substantial unittest test suite, a usage example, and a detailed explanation of race conditions. Be thorough and include executable code in every implementation and testing section.\n" + trailer, add_special_tokens=False)
    notes = []
    for i in range(200):
        notes.append(f"Requirement {i:03d}: jobs for tenant {i % 17} carry unique request identifiers, a payload revision {i + 100}, and an absolute deadline. The executor admits at most {4 + i % 9} pending operations per queue, records ordered results, propagates original exceptions, releases permits in finally blocks, and awaits cancelled tasks before shutdown. Retries must preserve the original request identity.\n")
    body = hf.encode("".join(notes), add_special_tokens=False)
    ids = prefix + body[:2842 - len(prefix) - len(suffix)] + suffix
    assert len(ids) == 2842
    text = hf.decode(ids, skip_special_tokens=False)
    save(ROOT / "prompt-ids.json", ids)
    (ROOT / "prompt.txt").write_text(text)
    return ids, hashlib.sha256(text.encode()).hexdigest()

def generate(generator, hf, ids, index):
    # Fixed-length throughput protocol: suppress EOS until the 2,048-token limit.
    job = Job(input_ids=torch.tensor([ids], dtype=torch.long), max_new_tokens=2048,
        min_new_tokens=2048, sampler=TopPSampler(.95, temperature=0), seed=20261026,
        stop_conditions=[166101], token_healing=False, stop_on_loop=None, decode_special_tokens=True)
    start = time.perf_counter()
    generator.enqueue(job)
    events = []
    output = []
    terminal = None
    while generator.num_remaining_jobs():
        for result in generator.iterate():
            elapsed = time.perf_counter() - start
            values = result.get("token_ids")
            if values is not None:
                tokens = values.reshape(-1).tolist()
                if tokens:
                    events.append({"seconds": elapsed, "token_ids": tokens})
                    output.extend(tokens)
            if result.get("eos"):
                terminal = {k: v for k, v in result.items() if isinstance(v, (str, int, float, bool, type(None)))}
                held = result.get("held", {}).get("token_ids")
                if held is not None:
                    tokens = held.reshape(-1).tolist()
                    if tokens:
                        events.append({"seconds": elapsed, "token_ids": tokens})
                        output.extend(tokens)
    torch.cuda.synchronize()
    wall = time.perf_counter() - start
    assert terminal and terminal["new_tokens"] == len(output) == 2048
    assert terminal["prompt_tokens"] == len(ids) == 2842
    assert terminal["cached_tokens"] == 0
    duration = events[-1]["seconds"] - events[0]["seconds"]
    first_n = len(events[0]["token_ids"])
    return {"index": index, "measured": index > 0, "input_tokens": len(ids),
        "output_tokens": len(output), "occupied_tokens": len(ids) + len(output),
        "output_token_ids": output, "text": hf.decode(output, skip_special_tokens=False),
        "events": events, "terminal": terminal, "ttft_ms": events[0]["seconds"] * 1000,
        "prefill_tokens_s": len(ids) / terminal["time_prefill"],
        "prompt_tokens_per_ttft_s": len(ids) / events[0]["seconds"],
        "output_tokens_s": (len(output) - first_n) / duration,
        "decode_interval_s": duration, "first_chunk_tokens": first_n,
        "wall_s": wall, "total_tokens_s": (len(ids) + len(output)) / wall}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=["fast", "max"], required=True)
    p.add_argument("--target-dir")
    p.add_argument("--draft-dir")
    p.add_argument("--repeats", type=int, default=3)
    a = p.parse_args()
    if a.preset == "max":
        os.environ["EXL3_QC_STAGING_PAGES"] = "64"
    else:
        os.environ.pop("EXL3_QC_STAGING_PAGES", None)
    is_fast = a.preset == "fast"
    assert a.repeats >= 1
    global TARGET, DRAFT
    from model_paths import resolve_models
    models = resolve_models(require_draft=True, target_dir=a.target_dir, draft_dir=a.draft_dir)
    TARGET = Path(models["target"]["path"])
    DRAFT = Path(models["draft"]["path"])
    capacity = 8192 if is_fast else 180224
    report = {"state": "loading", "preset": a.preset, "started_epoch": time.time(),
        "target": str(TARGET), "draft": str(DRAFT),
        "capacity": capacity, "weight_bpw": 3.55, "kv_bits": 0 if is_fast else 4,
        "draft_kv_bits": 0 if is_fast else 4, "sparse_draft_head": 16384,
        "batch_size": 1, "concurrency": 1, "temperature": 0, "thinking": False,
        "eos_policy": "suppressed until 2048 tokens for fixed-length throughput",
        "versions": {n: importlib.metadata.version(n) for n in ["exllamav3", "torch", "transformers"]},
        "gpu": torch.cuda.get_device_name(), "results": []}
    output_path = ROOT / (a.preset + ".json")
    save(output_path, report)
    hf = PreTrainedTokenizerFast.from_pretrained(str(TARGET), local_files_only=True)
    ids, digest = make_prompt(hf)
    report["prompt_sha256"] = digest
    if DRAFT is not None:
        import exllamav3.modules.sparse_head as sh
        profile_dir = ROOT / "sparse-profile"
        profile_dir.mkdir(exist_ok=True)
        source = PACKAGE / "profiles/sparse-seed.pt"
        if source.exists():
            destination = profile_dir / "49496f93c239aa4e.pt"
            shutil.copyfile(source, destination)
            report["sparse_profile_seed_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        sh.TOKEN_STATS_DIR = str(profile_dir)
    config = Config.from_directory(str(TARGET))
    model = Model.from_config(config)
    kwargs = {} if is_fast else {"layer_type": CacheLayer_quant, "k_bits": 4, "v_bits": 4}
    cache = Cache(model, max_num_tokens=capacity, **kwargs)
    draft = draft_cache = dc = None
    try:
        model.load(device=torch.device("cuda:0"), max_batch_size=1, max_chunk_size=2048, max_output_size=2048)
        tokenizer = Tokenizer.from_config(config)
        if DRAFT is not None:
            dc = Config.from_directory(str(DRAFT))
            draft = Model.from_config(dc)
            draft_cache = Cache(draft, max_num_tokens=capacity, **kwargs)
            draft.load(device=torch.device("cuda:0"), max_batch_size=1, max_chunk_size=2048, max_output_size=2048)
            draft.attach_to(model)
        report["state"] = "running"
        save(output_path, report)
        for index in range(a.repeats + 1):
            generator = Generator(model=model, cache=cache, tokenizer=tokenizer,
                draft_model=draft, draft_cache=draft_cache, num_draft_tokens=7,
                dynamic_draft_tokens=False, max_batch_size=1, max_chunk_size=2048,
                record_draft_stats=True, sparse_draft_head=16384)
            row = generate(generator, hf, ids, index)
            report["results"].append(row)
            save(output_path, report)
            print(json.dumps({k: row[k] for k in ["index", "measured", "input_tokens", "output_tokens", "output_tokens_s", "prefill_tokens_s", "ttft_ms"]}), flush=True)
            del generator
            gc.collect()
        measured = [r for r in report["results"] if r["measured"]]
        report["medians"] = {k: statistics.median(r[k] for r in measured) for k in ["output_tokens_s", "prefill_tokens_s", "prompt_tokens_per_ttft_s", "ttft_ms", "wall_s", "total_tokens_s"]}
        report["output_rate_range"] = [min(r["output_tokens_s"] for r in measured), max(r["output_tokens_s"] for r in measured)]
        report["state"] = "complete"
        report["completed_epoch"] = time.time()
        save(output_path, report)
        print("COMPLETE", a.preset, report["medians"], flush=True)
    except Exception as e:
        report["state"] = "failed"
        report["error"] = repr(e)
        save(output_path, report)
        raise
    finally:
        if draft is not None:
            draft.unload()
            dc.stc.close()
        model.unload()
        config.stc.close()

if __name__ == "__main__":
    main()
