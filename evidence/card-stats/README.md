# Nanbeige stats matching the supplied card's token counts

Fresh RTX 5090 measurements: **116.9 output tok/s** with the fast preset, or **70.4 output tok/s** with the maximum-context preset. Both use the existing Nanbeige4.3-33B EXL3 3.55bpw target. Same input/output lengths as the supplied card; its exact prompt was unavailable.

## Stats card comparison

Batch 1, concurrency 1, **2,842 input + 2,048 output = 4,890 occupied tokens**. Zero cached/reused prompt tokens. Local results are medians of three runs, after one full-length excluded warmup per profile.

| Stat | Supplied card | RTX 5090 fast | RTX 5090 max capacity |
|---|---:|---:|---:|
| Output speed, tok/s | 125.1 | 116.9 | 70.4 |
| Native prefill, tok/s | 6,810 reported | 3,130 | 3,157 |
| TTFT, ms | 417 | 926 | 917 |
| Context limit/capacity, tokens | 524,288 | 8,192 | 480,256 |
| Actual occupied tokens | 4,890 | 4,890 | 4,890 |
| Weight precision | FP8 | EXL3 3.55bpw | EXL3 3.55bpw |
| KV precision | Not specified | FP16 | 2-bit |
| Speculative method | ngram_gpu | DFlash2, sparse16K | Off |
| Engine | vLLM | ExLlamaV3 1.6.5 | ExLlamaV3 1.6.5 |
| Hardware | RTX PRO 6000 Blackwell Max-… | RTX 5090 | RTX 5090 |

The source hardware label is truncated in the screenshot. Its complete run record, exact prompt, flags and timing implementation could not be fetched (public API GET returned 403). Its values are a transcription of the user-provided screenshot supplied in the session (transcribed comparison values are saved in stats.json). This compares token counts and reported metrics across different hardware, precision, prompts and speculative methods; it is not an exact replication or quality-parity test.

**Context capacity and actual occupancy are separate.** The supplied card's 524,288 context label does not mean it read 524,288 tokens. The prior near-full run really used 479,744 input + 256 output, and measured 23.2 output tok/s after a 16.7-minute first-token wait. The same maximum-context preset here measures 70.4 tok/s with only 4,890 tokens occupied.

## Protocol and definitions

Fixed synthetic engineering brief, exact 2,842 IDs saved in prompt-ids.json, with the model's non-thinking chat framing. Greedy temperature 0, top-p 0.95. For this fixed-length performance test, EOS is suppressed until exactly 2,048 tokens; this differs from natural-completion quality testing. No output is claimed to be a finished or validated implementation.

Each profile received one full-length excluded warmup followed by three measured requests. Every request used a fresh generator/page table, and native metadata confirms zero cached prompt tokens. Sparse draft frequency statistics began as a private copy of the existing launcher's profile, learned during the run and stayed inside this evidence directory; the user's profile was not overwritten. The three measured outputs were identical within each greedy profile.

Output throughput excludes the first streamed chunk: `(output_count - first_chunk_count) / (last_arrival_time - first_arrival_time)`. Native prefill is `input_count / native_time_prefill`. TTFT is enqueue to first streamed token with a resident warmed model, excluding initial model loading and CPU tokenization. A conservative `input_count / TTFT` is also saved: **3,068 / 3,100 tok/s** for fast / max. The screenshot's precise prefill timing formula is unknown, so do not assume it is identical to the native counter.

Per-request output ranges: fast 116.568–117.104 tok/s; max 70.364–71.734. These are three repeated greedy requests on one prompt, not a broad prompt suite. The earlier 103.4 tok/s median covered five natural prompt types with short inputs and a 384-token output cap.

Hardware: RTX 5090, Ryzen 9 9950X3D, about 128 GB RAM, running desktop. Existing 600 W power limit retained. Runtime ExLlamaV3 1.6.5+cu132.torch2.13.0, torch 2.13.0+cu132, Transformers 4.57.6. All weights remain in the default Hugging Face cache; no downloads, package changes or installed engine/Strata source changes.

Fast profile: 8,192 capacity, FP16 target/draft KV, native DFlash2, 7 proposals, sparse draft head 16K, 2,048-token chunks, CUDA graphs. Maximum profile: 480,256 capacity, 2-bit target KV, no draft, 2,048-token chunks, EXL3_QC_STAGING_PAGES=64. The maximum profile allocates its full configured KV pool even though the actual request is short.

Reference methodology and field definitions: [LocalMaxxing API docs](https://localmaxxing.com/en/api-docs). The site defines prefillTokens as tokens already held in KV before the run; zero means no already-cached prefix, not an absence of KV storage.

## Reproduce

From ${HOME}/Work/Strata, run profiles sequentially:

```sh
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 EXL3_GRAPHS=1 \
  ${HOME}/.venvs/nanbeige-exl3-165/bin/python -u \
  build-nanbeige-evidence/2026-10-26-card-stats/bench.py --preset fast

OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 EXL3_GRAPHS=1 EXL3_QC_STAGING_PAGES=64 \
  ${HOME}/.venvs/nanbeige-exl3-165/bin/python -u \
  build-nanbeige-evidence/2026-10-26-card-stats/bench.py --preset max

python build-nanbeige-evidence/2026-10-26-card-stats/audit.py
```

The measured executions used user services nanbeige-card-fast / nanbeige-card-max, each capped at 64 GiB process-group RAM with swap disabled. Both were observed inactive/exited 0. Telemetry stopped, GPU returned to desktop use. Presets and previous raw benchmarks remain available. No external result submission was authorized or performed.

## Evidence

- fast.json / max.json: raw per-token IDs/times, output text, native completion metadata and summaries.
- prompt-ids.json / prompt.txt: exact common input and hash.
- stats-audit.json: independently recomputed counts, zero prefix reuse, metrics, medians and repeated-output consistency.
- stats.json / stats.csv: compact metric records.
- gpu-telemetry.csv / resource-summary.json: sampled resource data, including load and warmup.
- reference-card.png / reference.json: user-supplied image and transcribed comparison record.
- fast.log / max.log: all request summaries and completion status.
- completion-audit.json: file hashes, observed exit state and final cleanup receipt.

Documentation correction discovered during this comparison: the previous short speed suite's raw input IDs/native metadata show **89–103 input tokens (473–487 occupied)**. Earlier prose said 85–99 (469–483); that count typo was corrected in the reports/project state. Its raw results and measured rates are unchanged.
