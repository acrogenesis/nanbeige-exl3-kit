"""Single-stream Nanbeige benchmark using stock ExLlamaV3 APIs."""
import argparse
import gc
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import statistics
import subprocess
import time

os.environ.setdefault("EXL3_GRAPHS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


import torch
from exllamav3 import Config, Model, Cache, CacheLayer_quant, Tokenizer, Generator, Job, TopPSampler
from transformers import PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parent
PROMPTS = [
    {"name": "code", "prompt": "Write a production-quality Python implementation of a bounded asynchronous worker pool. It must accept async callables, limit concurrency, preserve submission order in returned results, propagate exceptions, and cancel outstanding work when the caller is cancelled. Explain the cancellation behavior and show a usage example."},
    {"name": "debugging", "prompt": "A PostgreSQL-backed web service sometimes charges a customer twice when a request times out and the client retries. The payment provider can take 30 seconds to reply. Design an idempotent payment workflow, including the database schema, transaction boundaries, recovery after a crash, and handling concurrent retries. Give concrete pseudocode and discuss failure cases."},
    {"name": "math", "prompt": "Derive the mean and variance of a geometric random variable that counts the number of independent Bernoulli trials up to and including the first success. Use two different derivations, justify the infinite-series manipulations, and work a numerical example with success probability 0.2. Explain common off-by-one mistakes."},
    {"name": "explanation", "prompt": "Explain how a database B-tree index works to a programmer who understands arrays and hash tables. Walk through inserting keys, splitting nodes, range scans, and deleting a key. Compare clustered and secondary indexes, and describe why an index can make writes slower. Use a concrete small example."},
    {"name": "chinese", "prompt": "请详细设计一个在线书店的库存预留系统。要求处理并发下单、支付超时、重复回调、订单取消和服务崩溃后的恢复。请给出数据库表结构、关键事务的伪代码，以及一个包含两个并发订单的具体例子，并解释如何保证不会超卖。"},
]

def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")

def run_case(generator, hf, prompt, *, temperature, thinking, limit, seed, measured, on_text=None):
    ids = hf.apply_chat_template(
        [{"role": "user", "content": prompt["prompt"]}], tokenize=True,
        add_generation_prompt=True, enable_thinking=thinking, preserve_thinking=False,
    )
    job = Job(input_ids=torch.tensor([ids], dtype=torch.long), max_new_tokens=limit,
              sampler=TopPSampler(0.95, temperature=temperature), seed=seed,
              stop_conditions=[166101], token_healing=False, stop_on_loop=None,
              decode_special_tokens=True)
    output = []
    events = []
    terminal = None
    start = time.perf_counter()
    generator.enqueue(job)
    while generator.num_remaining_jobs():
        for result in generator.iterate():
            now = time.perf_counter()
            values = result.get("token_ids")
            if values is not None:
                tokens = values.reshape(-1).tolist()
                output.extend(tokens)
                if tokens:
                    events.append({"seconds": now - start, "token_ids": tokens})
            if on_text is not None and result.get("text"):
                on_text(result["text"])
            if result.get("eos"):
                terminal = {k: v for k, v in result.items()
                            if k not in ("job", "token_ids", "held")
                            and isinstance(v, (str, int, float, bool, type(None)))}
                held = result.get("held", {}).get("token_ids")
                if held is not None:
                    tokens = held.reshape(-1).tolist()
                    output.extend(tokens)
                    if tokens:
                        events.append({"seconds": now - start, "token_ids": tokens})
    torch.cuda.synchronize()
    wall = time.perf_counter() - start
    assert terminal is not None, "Missing terminal result"
    assert len(output) == terminal["new_tokens"] == job.new_tokens, (len(output), terminal, job.new_tokens)
    assert terminal.get("cached_tokens", 0) == 0, terminal
    # For multi-token speculative chunks, exclude the whole initial chunk.
    # Count only tokens actually emitted between the first and last arrivals.
    first_n = len(events[0]["token_ids"])
    decode_s = events[-1]["seconds"] - events[0]["seconds"]
    row = {
        "name": prompt["name"], "measured": measured, "seed": seed,
        "temperature": temperature, "top_p": 0.95, "thinking": thinking,
        "input_token_ids": ids, "prompt_tokens": len(ids), "output_tokens": len(output),
        "output_token_ids": output, "text": hf.decode(output, skip_special_tokens=False),
        "events": events, "ttft_s": events[0]["seconds"], "wall_s": wall,
        "decode_s": decode_s, "decode_tokens_s": (len(output) - first_n) / decode_s if decode_s > 0 else None,
        "end_to_end_tokens_s": len(output) / wall,
        "terminal": terminal, "draft_stats": getattr(job, "draft_stats", []),
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    sh = getattr(generator, "sparse_head", None)
    if sh is not None:
        row["sparse_head_stats"] = {k: v for k, v in vars(sh).items()
                                    if isinstance(v, (str, int, float, bool, type(None)))}
    return row

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["plain", "draft"], required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--target-dir")
    parser.add_argument("--draft-dir")
    parser.add_argument("--cache-tokens", type=int, default=8192)
    parser.add_argument("--kv-bits", type=int, choices=[0, 8], default=8,
                        help="0 selects the default FP16 cache; 8 selects Q8 K/V")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-tokens", type=int, default=384)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--proposals", type=int, default=7)
    parser.add_argument("--sparse-draft-head", type=int, default=0)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    if args.sparse_draft_head:
        import exllamav3.modules.sparse_head as sparse_head_module
        sparse_head_module.TOKEN_STATS_DIR = str(ROOT / (args.label + "-token-stats"))
    from model_paths import resolve_models
    models = resolve_models(require_draft=args.mode == "draft", target_dir=args.target_dir, draft_dir=args.draft_dir)
    target_dir = args.target_dir or models["target"]["path"]
    draft_dir = args.draft_dir or models["draft"]["path"]
    prompts = [p for p in PROMPTS if not args.only or p["name"] in args.only]
    report = {"settings": vars(args), "target_dir": target_dir, "draft_dir": draft_dir,
              "versions": {p: importlib.metadata.version(p) for p in ["torch", "exllamav3", "transformers"]},
              "platform": platform.platform(), "gpu": torch.cuda.get_device_name(),
              "started_epoch": time.time(), "state": "loading", "results": []}
    save(ROOT / (args.label + ".json"), report)
    print("Loading target", target_dir, flush=True)
    config = Config.from_directory(target_dir)
    config.override_dynamic_seq_len(2048)
    model = Model.from_config(config)
    cache_kwargs = {"layer_type": CacheLayer_quant, "k_bits": 8, "v_bits": 8} if args.kv_bits else {}
    cache = Cache(model, max_num_tokens=args.cache_tokens, **cache_kwargs)
    model.load(device=torch.device("cuda:0"), max_batch_size=1, max_output_size=2048)
    tokenizer = Tokenizer.from_config(config)
    hf = PreTrainedTokenizerFast.from_pretrained(target_dir, local_files_only=True)
    draft = draft_config = draft_cache = None
    if args.mode == "draft":
        print("Loading draft", draft_dir, flush=True)
        draft_config = Config.from_directory(draft_dir)
        draft_config.override_dynamic_seq_len(2048)
        draft = Model.from_config(draft_config)
        draft_cache = Cache(draft, max_num_tokens=args.cache_tokens)
        draft.load(device=torch.device("cuda:0"), max_batch_size=1, max_output_size=2048)
        draft.attach_to(model)
        assert draft_config.tap_shift == 0 and draft_config.mask_token_id == 0
    try:
        report["state"] = "running"
        for measured in [False, True]:
            for repetition in range(args.repeats if measured else 1):
                for prompt in prompts:
                    # A new page table per request ensures no warm prefix reuse.
                    generator = Generator(model=model, cache=cache, tokenizer=tokenizer,
                        draft_model=draft, draft_cache=draft_cache,
                        num_draft_tokens=args.proposals, dynamic_draft_tokens=False,
                        max_batch_size=1, max_chunk_size=2048, record_draft_stats=True,
                        **({"sparse_draft_head": args.sparse_draft_head} if args.sparse_draft_head else {}))
                    row = run_case(generator, hf, prompt, temperature=args.temperature,
                                   thinking=args.thinking, limit=args.output_tokens,
                                   seed=20261025 + repetition, measured=measured)
                    report["results"].append(row)
                    save(ROOT / (args.label + ".json"), report)
                    print(json.dumps({k: row[k] for k in ["name", "measured", "output_tokens", "ttft_s", "decode_tokens_s", "end_to_end_tokens_s"]}), flush=True)
                    del generator
                    gc.collect()
        rows = [r for r in report["results"] if r["measured"]]
        report["median_decode_tokens_s"] = statistics.median(r["decode_tokens_s"] for r in rows)
        report["pooled_decode_tokens_s"] = sum(r["output_tokens"] - len(r["events"][0]["token_ids"]) for r in rows) / sum(r["decode_s"] for r in rows)
        report["by_prompt"] = {p["name"]: statistics.median(r["decode_tokens_s"] for r in rows if r["name"] == p["name"]) for p in prompts}
        report["state"] = "complete"
        report["completed_epoch"] = time.time()
        save(ROOT / (args.label + ".json"), report)
        print("COMPLETE", args.label, report["median_decode_tokens_s"], report["by_prompt"], flush=True)
    finally:
        if draft is not None:
            draft.unload()
            draft_config.stc.close()
        model.unload()
        config.stc.close()

if __name__ == "__main__":
    main()
