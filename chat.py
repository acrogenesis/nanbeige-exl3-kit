"""One prompt, streaming output; uses the same models and API as the benchmark."""
import argparse
import json
import sys

from bench_exl3 import ROOT, Config, Model, Cache, CacheLayer_quant, Tokenizer, Generator, PreTrainedTokenizerFast, run_case, torch

def main():
    p = argparse.ArgumentParser()
    p.add_argument("prompt", nargs="?")
    p.add_argument("--prompt-file")
    p.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--cache-tokens", type=int, default=8192)
    p.add_argument("--kv-bits", type=int, choices=[0, 2, 3, 4, 5, 6, 7, 8], default=8)
    p.add_argument("--draft-kv-bits", type=int, choices=[0, 2, 3, 4, 5, 6, 7, 8], default=0)
    p.add_argument("--no-draft", action="store_true")
    p.add_argument("--proposals", type=int, default=7)
    p.add_argument("--sparse-draft-head", type=int, default=0)
    p.add_argument("--target-dir")
    p.add_argument("--output-json")
    a = p.parse_args()
    if bool(a.prompt) == bool(a.prompt_file):
        p.error("Pass either a prompt or --prompt-file.")
    if a.prompt_file:
        from pathlib import Path
        a.prompt = Path(a.prompt_file).read_text()
    from model_paths import resolve_models
    models = resolve_models(require_draft=not a.no_draft, target_dir=a.target_dir)
    target_dir = a.target_dir or models["target"]["path"]
    config = Config.from_directory(target_dir)
    config.override_dynamic_seq_len(2048)
    if a.cache_tokens <= 0 or a.cache_tokens % 256:
        p.error("--cache-tokens must be a positive multiple of 256.")
    if a.cache_tokens > config.max_position_embeddings:
        p.error(f"The model supports at most {config.max_position_embeddings} context tokens.")
    hf = PreTrainedTokenizerFast.from_pretrained(target_dir, local_files_only=True)
    input_ids = hf.apply_chat_template([{"role": "user", "content": a.prompt}],
        tokenize=True, add_generation_prompt=True, enable_thinking=a.thinking, preserve_thinking=False)
    reserve = a.max_new_tokens + 256 + (0 if a.no_draft else a.proposals)
    if len(input_ids) + reserve > a.cache_tokens:
        p.error(f"Input has {len(input_ids)} tokens; with {reserve} reserved for output and bookkeeping, "
                f"it exceeds the {a.cache_tokens}-token context. Shorten the input or output limit.")
    print(f"Input {len(input_ids)} tokens; context capacity {a.cache_tokens}; output cap {a.max_new_tokens}.", file=sys.stderr)
    model = Model.from_config(config)
    cache_kwargs = {"layer_type": CacheLayer_quant, "k_bits": a.kv_bits, "v_bits": a.kv_bits} if a.kv_bits else {}
    cache = Cache(model, max_num_tokens=a.cache_tokens, **cache_kwargs)
    print("Loading Nanbeige target...", file=sys.stderr, flush=True)
    model.load(device=torch.device("cuda:0"), max_batch_size=1, max_output_size=2048)
    tokenizer = Tokenizer.from_config(config)
    draft = draft_cache = dc = None
    if not a.no_draft:
        dc = Config.from_directory(models["draft"]["path"])
        dc.override_dynamic_seq_len(2048)
        draft = Model.from_config(dc)
        dk = {"layer_type": CacheLayer_quant, "k_bits": a.draft_kv_bits, "v_bits": a.draft_kv_bits} if a.draft_kv_bits else {}
        draft_cache = Cache(draft, max_num_tokens=a.cache_tokens, **dk)
        draft.load(device=torch.device("cuda:0"), max_batch_size=1, max_output_size=2048)
        draft.attach_to(model)
        if a.sparse_draft_head:
            from model_paths import seed_sparse_profile
            seed_sparse_profile(target_dir, config.vocab_size)
    try:
        generator = Generator(model=model, cache=cache, tokenizer=tokenizer,
            draft_model=draft, draft_cache=draft_cache, num_draft_tokens=a.proposals,
            dynamic_draft_tokens=False, max_batch_size=1, max_chunk_size=2048,
            record_draft_stats=True,
            **({"sparse_draft_head": a.sparse_draft_head} if a.sparse_draft_head and draft is not None else {}))
        row = run_case(generator, hf, {"name": "chat", "prompt": a.prompt},
                       temperature=a.temperature, thinking=a.thinking,
                       limit=a.max_new_tokens, seed=20261025, measured=False,
                       on_text=lambda text: print(text, end="", flush=True))
        print()
        rate = f"{row['decode_tokens_s']:.2f}" if row['decode_tokens_s'] is not None else "unmeasurable (one chunk)"
        print(f"\n{row['output_tokens']} tokens, {rate} decode tokens/s; "
              f"first token {row['ttft_s']:.3f}s (includes first-use compilation).", file=sys.stderr)
        if a.output_json:
            from pathlib import Path
            Path(a.output_json).write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n")
    finally:
        if draft is not None:
            draft.unload()
            dc.stc.close()
        model.unload()
        config.stc.close()

if __name__ == "__main__":
    main()
