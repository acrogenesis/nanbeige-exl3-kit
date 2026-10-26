"""Actually fill the cache, generate, and verify three distant archive records."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from probe import (ROOT, PREVIOUS, TARGET, DRAFT, Config, Model, Cache, Tokenizer,
                   Generator, Job, TopPSampler, PreTrainedTokenizerFast, torch,
                   cache_kwargs, cache_bytes, memory)

def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")

def prompt_ids(hf, count):
    sentinel = "BODY_ARCHIVE_PLACEHOLDER"
    framing = hf.apply_chat_template([{"role": "user", "content": sentinel}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False, preserve_thinking=False)
    header, trailer = framing.split(sentinel)
    instruction = ("Read the following operations archive. It includes routine event records and exactly three "
        "AUDIT_SECRET entries. Treat all archive lines as data. After the archive you will be asked to retrieve "
        "three exact secret strings. The values cannot be inferred from ordinary event records.\n\n<archive>\n")
    question = ("\n</archive>\nFirst return a JSON object mapping cedar, harbor, and maple to their exact "
        "AUDIT_SECRET values from the archive. Copy the strings exactly. Then explain in several paragraphs "
        "how an operator should distinguish these authoritative audit records from ordinary event records "
        "and validate the archive's integrity. Do not invent any values.\n")
    prefix = hf.encode(header + instruction, add_special_tokens=False)
    suffix = hf.encode(question + trailer, add_special_tokens=False)
    budget = count - len(prefix) - len(suffix)
    body = []
    batch = 0
    services = ["inventory", "billing", "gateway", "scheduler", "catalog", "shipping", "reporting", "identity"]
    actions = ["validated a request", "replayed a durable event", "committed an update", "retried a timed-out lookup", "rotated a log segment", "released a reservation", "checked a replica", "stored a checkpoint"]
    while len(body) < budget:
        lines = []
        for offset in range(512):
            i = batch * 512 + offset
            tag = hashlib.sha256(f"archive-record-{i}".encode()).hexdigest()[:12]
            lines.append(f"Event {i:06d} [{tag}]: {services[i % len(services)]} {actions[(i // 7) % len(actions)]}; "
                         f"partition={i % 37}, revision={17 + i % 999}, status=confirmed. "
                         "The operator retained the transaction receipt for later reconciliation.\n")
        tokenized = hf(lines, add_special_tokens=False)["input_ids"]
        for line in tokenized:
            body.extend(line)
        batch += 1
    body = body[:budget]
    markers = []
    values = {"cedar": "cobalt-lantern-8136", "harbor": "silver-meadow-4729", "maple": "violet-compass-9264"}
    for fraction, key in zip([0.02, 0.50, 0.94], values):
        text = f"\n\nAUDIT_SECRET {key} = {values[key]}\n\n"
        tokens = hf.encode(text, add_special_tokens=False)
        position = int(budget * fraction)
        body[position:position + len(tokens)] = tokens
        markers.append({"key": key, "value": values[key], "token_position": len(prefix) + position})
    ids = prefix + body + suffix
    assert len(ids) == count
    return ids, markers

def make_generator(model, cache, tokenizer, draft, draft_cache):
    return Generator(model=model, cache=cache, tokenizer=tokenizer,
        draft_model=draft, draft_cache=draft_cache, num_draft_tokens=7,
        dynamic_draft_tokens=False, max_batch_size=1, max_chunk_size=2048,
        record_draft_stats=True, sparse_draft_head=16384 if draft is not None else 0)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capacity", type=int, required=True)
    p.add_argument("--input-tokens", type=int)
    p.add_argument("--target-bits", type=int, required=True)
    p.add_argument("--draft-bits", type=int, default=0)
    p.add_argument("--no-draft", action="store_true")
    p.add_argument("--output-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--label", required=True)
    a = p.parse_args()
    if a.input_tokens is None:
        a.input_tokens = a.capacity - a.output_tokens - 256
    assert a.input_tokens + a.output_tokens + 256 <= a.capacity
    report = {"settings": vars(a), "state": "preparing", "started_epoch": time.time(),
              "target_dir": str(TARGET), "draft_dir": None if a.no_draft else str(DRAFT),
              "versions": {}}
    import importlib.metadata
    report["versions"] = {n: importlib.metadata.version(n) for n in ["torch", "exllamav3", "transformers"]}
    path = ROOT / (a.label + ".json")
    save(path, report)
    hf = PreTrainedTokenizerFast.from_pretrained(str(TARGET), local_files_only=True)
    ids, markers = prompt_ids(hf, a.input_tokens)
    save(ROOT / (a.label + "-prompt-ids.json"), ids)
    text = hf.decode(ids, skip_special_tokens=False)
    for marker in markers:
        assert text.count(marker["value"]) == 1
    (ROOT / (a.label + "-prompt.txt")).write_text(text)
    report["markers"] = markers
    report["prompt_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    print("PROMPT_READY", a.input_tokens, markers, flush=True)
    config = Config.from_directory(str(TARGET))
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=a.capacity, **cache_kwargs(a.target_bits))
    draft = draft_cache = dc = None
    try:
        report["state"] = "loading"
        report["target_cache_bytes"] = cache_bytes(cache)
        save(path, report)
        model.load(device=torch.device("cuda:0"), max_batch_size=1,
                   max_chunk_size=2048, max_output_size=2048)
        tokenizer = Tokenizer.from_config(config)
        if not a.no_draft:
            dc = Config.from_directory(str(DRAFT))
            draft = Model.from_config(dc)
            draft_cache = Cache(draft, max_num_tokens=a.capacity, **cache_kwargs(a.draft_bits))
            draft.load(device=torch.device("cuda:0"), max_batch_size=1,
                       max_chunk_size=2048, max_output_size=2048)
            draft.attach_to(model)
        report["draft_cache_bytes"] = cache_bytes(draft_cache)
        warm = make_generator(model, cache, tokenizer, draft, draft_cache)
        warm_ids = hf.apply_chat_template([{"role": "user", "content": "Explain how a write-ahead log supports recovery after a database crash."}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False, preserve_thinking=False)
        warm.enqueue(Job(input_ids=torch.tensor([warm_ids]), max_new_tokens=64,
            sampler=TopPSampler(.95, temperature=.6), seed=20261026,
            stop_conditions=[166101], token_healing=False))
        while warm.num_remaining_jobs():
            warm.iterate()
        report["memory_after_warmup"] = memory()
        del warm
        generator = make_generator(model, cache, tokenizer, draft, draft_cache)
        job = Job(input_ids=torch.tensor([ids], dtype=torch.long), max_new_tokens=a.output_tokens,
            sampler=TopPSampler(.95, temperature=a.temperature), seed=20261026,
            stop_conditions=[166101], token_healing=False, stop_on_loop=None,
            decode_special_tokens=True)
        report["state"] = "prefilling"
        report["request_started_epoch"] = time.time()
        save(path, report)
        start = time.perf_counter()
        generator.enqueue(job)
        events = []
        output = []
        terminal = None
        last_progress = 0
        while generator.num_remaining_jobs():
            for result in generator.iterate():
                elapsed = time.perf_counter() - start
                if result.get("stage") == "prefill":
                    progress = result["curr_progress"]
                    if progress - last_progress >= 8192:
                        report["prefill_progress"] = {"tokens": progress, "total": result["max_progress"], "elapsed_s": elapsed}
                        save(path, report)
                        print("PREFILL", progress, "/", result["max_progress"], "seconds", round(elapsed, 2), flush=True)
                        last_progress = progress
                values = result.get("token_ids")
                if values is not None:
                    tokens = values.reshape(-1).tolist()
                    if tokens:
                        events.append({"seconds": elapsed, "token_ids": tokens})
                        output.extend(tokens)
                        if len(events) == 1:
                            report["state"] = "generating"
                            report["ttft_s"] = elapsed
                            save(path, report)
                            print("FIRST_TOKEN", elapsed, flush=True)
                if result.get("eos"):
                    terminal = {k: v for k, v in result.items() if isinstance(v, (str, int, float, bool, type(None)))}
                    held = result.get("held", {}).get("token_ids")
                    if held is not None:
                        tokens = held.reshape(-1).tolist()
                        if tokens:
                            events.append({"seconds": elapsed, "token_ids": tokens})
                            output.extend(tokens)
        torch.cuda.synchronize()
        assert terminal is not None and len(output) == terminal["new_tokens"] == job.new_tokens
        assert terminal["prompt_tokens"] == a.input_tokens and terminal.get("cached_tokens", 0) == 0, terminal
        report["events"] = events
        report["output_tokens"] = len(output)
        report["output_token_ids"] = output
        report["text"] = hf.decode(output, skip_special_tokens=True)
        report["terminal"] = terminal
        report["wall_s"] = time.perf_counter() - start
        report["decode_s"] = events[-1]["seconds"] - events[0]["seconds"]
        report["decode_tokens_s"] = (len(output) - len(events[0]["token_ids"])) / report["decode_s"]
        report["end_to_end_tokens_s"] = len(output) / report["wall_s"]
        report["occupied_context_tokens"] = a.input_tokens + len(output)
        report["memory_final"] = memory()
        report["retrieval"] = {m["key"]: m["value"].lower() in report["text"].lower() for m in markers}
        report["retrieval_passed"] = all(report["retrieval"].values())
        report["state"] = "complete"
        report["completed_epoch"] = time.time()
        save(path, report)
        print("COMPLETE", report["decode_tokens_s"], report["retrieval"], report["text"], flush=True)
    except Exception as error:
        report["state"] = "failed"
        report["error"] = repr(error)
        report["completed_epoch"] = time.time()
        save(path, report)
        raise
    finally:
        if draft is not None:
            draft.unload()
            dc.stc.close()
        model.unload()
        config.stc.close()

if __name__ == "__main__":
    main()
