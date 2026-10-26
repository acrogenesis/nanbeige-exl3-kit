# Nanbeige maximum context on the RTX 5090

The same Nanbeige4.3-33B 3.55bpw EXL3 model passed a real near-full **480,256-token capacity** test. The request contained **479,744 input tokens and 256 generated tokens**, occupying 480,000 tokens. All three distant key/value pairs were retrieved exactly. No prompt tokens were reused from cache.

The selected preset uses **2-bit keys and values, with DFlash2 disabled**. The weight revision and weight precision are unchanged. This is the largest capacity validated here with the desktop running. It is a separate preset from the earlier short-context 100 tok/s run.

## Measured result

Hardware: one NVIDIA RTX 5090, 32,607 MiB reported VRAM, Ryzen 9 9950X3D, about 128 GB system RAM. Desktop remained running; GPU power limit stayed at its existing 600 W. Runtime: ExLlamaV3 1.6.5+cu132.torch2.13.0, PyTorch 2.13.0+cu132, Transformers 4.57.6. No Strata engine or installed runtime source was modified.

| Measurement | Result |
|---|---:|
| Configured context capacity | 480,256 tokens |
| Actual input | 479,744 tokens |
| Actual generated output | 256 tokens |
| Actual occupied context | 480,000 tokens |
| First token / complete initial read | 1,001.822 s (16.697 min) |
| Mean prompt processing rate | 478.897 tokens/s |
| Decode after first token | 23.200 tokens/s |
| Request wall time | 1,012.814 s |
| Output tokens / complete request wall time | 0.253 tokens/s |
| Exact key/value retrieval | 3 of 3 passed |
| Reused prompt tokens | 0 |
| Target KV allocation | 14.885 GiB |
| Peak sampled total GPU memory | 32,171 MiB |
| CUDA free memory at final snapshot | 157.25 MiB |

Decode is `(256 - 1) / (last token time - first token time)`. It excludes the initial read. It is a single request, not a multi-run median. Generation stopped at the requested 256-token cap after returning complete JSON and part of the explanatory prose. The previous 103.4 tok/s median used 89–103 input tokens, FP16 KV and DFlash2; it does not apply to this nearly 480K request.

## Retrieval and input integrity

A synthetic operations archive contains unique hashed event lines plus three audit records. Their values appear exactly once and are absent from the final question. The record starts are at input token positions 9,672, 239,873 and 450,889, approximately 2%, 50% and 94% through the prompt.

The generated JSON was:

```json
{
  "cedar": "cobalt-lantern-8136",
  "harbor": "silver-meadow-4729",
  "maple": "violet-compass-9264"
}
```

`audit_retrieval.py` parses the JSON and checks exact key-to-value bindings. It also checks the saved prompt hash, raw input/output counts, streamed token IDs, native completion metadata, zero cached prompt tokens and independently recomputed decode timing. This is one synthetic retrieval check, not a comprehensive quality evaluation of 2-bit KV or parity against BF16.

## Capacity selection

Short allocation probes used the same target plus native DFlash2 and two 128-token generations. The following capacities passed with at least about 1 GiB free after that short workload. These rows are allocation/workspace checks, not occupied long-context benchmarks:

| Target and draft KV precision | Capacity with DFlash2 |
|---|---:|
| FP16 | 58,624 |
| 8-bit | 109,312 |
| 4-bit | 209,152 |
| 2-bit | 372,224 |

Removing the drafter while retaining the target passed target-only 2-bit probes at 450,560 and 474,880 tokens. The latter left 1.181 GiB after short generation. The selected 480,256 capacity was then actually filled and generated successfully. Full prefill uses additional workspace, so the short-probe 1 GiB reserve is not the final free memory: only 157.25 MiB remained in the final CUDA snapshot. Additional desktop GPU use may require lowering `--cache-tokens` in multiples of 256.

The model's stated window is 524,288 tokens. That capacity was not executed here. Exact cache geometry and observed active memory indicate insufficient room for the additional cache plus working buffers under this configuration. This is a memory-based estimate, not an observed 512K OOM. User explicitly chose to retain the existing 3.55bpw weights rather than use stronger weight quantization to pursue the entire window. No new weights were downloaded.

## Run the preset

From `${HOME}/Work/Strata`:

```sh
./build-nanbeige-evidence/2026-10-26-max-context/run-max.sh \
  --prompt-file /path/to/document.txt --max-new-tokens 256
```

A positional prompt also works. The file is plain user content; the launcher applies the model's chat template. The capacity includes input, output and a 256-token bookkeeping reserve. Default output cap is 1,024 tokens. The launcher counts input and rejects an oversized request before allocating the model. Each invocation loads, streams one completion and exits; no server is left running.

The launcher sets `EXL3_QC_STAGING_PAGES=64`, a 16K-token prefill staging buffer for compressed KV. This bounds transient scratch and still attends to older cache slices; it does not truncate the context. It uses batch size 1, 2,048-token chunks, CUDA graphs, thinking off, temperature 0.6 and top-p 0.95 by default.

The existing short-context preset remains available:

```sh
./build-nanbeige-evidence/2026-10-25-100tps/run-fast.sh "your prompt"
```

Both launchers passed a fresh actual completion for `17 * 23`, returning exactly `391` and stopping on the native end token. The maximum-context launcher was tested through `--prompt-file`. Shared ignored `chat.py` now accepts prompt files, 2–8-bit KV options and a no-draft mode, and validates context occupancy before GPU loading. This is an external ExLlamaV3 launcher, not a Nanbeige port into Strata's own engine.

## Reproduce and inspect

Full run used the isolated existing Python environment, the same target snapshot, and these settings:

```sh
EXL3_GRAPHS=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 EXL3_QC_STAGING_PAGES=64 \
  ${HOME}/.venvs/nanbeige-exl3-165/bin/python -u \
  build-nanbeige-evidence/2026-10-26-max-context/full_context.py \
  --capacity 480256 --target-bits 2 --no-draft \
  --label full480k-b2-plain --output-tokens 256 --temperature 0.6
```

Executed under user service `nanbeige-context-full480k.service` with 64 GiB process-group RAM limit and swap disabled. One 64-token warmup was excluded; the measured request used a fresh generator/page table. Its service was observed inactive/exited 0. No GPU inference jobs remain, telemetry is stopped, and GPU memory returned to desktop use (1,163 MiB at cleanup).

Weights remain in the default Hugging Face cache:
`~/.cache/huggingface/hub/models--ProCreations--Nanbeige4.3-33B-EXL3/snapshots/56a23706262458b23ba7dde5ba1eea053a109544`.

Artifacts:

- `full480k-b2-plain.json`: all output IDs, event times, native metadata and memory snapshots.
- `full480k-b2-plain-prompt-ids.json` and `full480k-b2-plain-prompt.txt`: exact measured input and chat framing.
- `full480k-b2-plain-audit.json`: exact retrieval, count, hash and timing audit.
- `full480k.log` and `prefill-progress.csv`: native prompt progress.
- `gpu-telemetry.csv` and `resource-summary.json`: sampled GPU resource data.
- `allocation-results.json`, `allocation-winners.json`, individual `probe-*.json` and logs: observed capacity checks.
- `capacity-plan.json`: estimates from exact per-token cache geometry; estimates are separate from observed allocations.
- `launcher-max-smoke.*` and `launcher-fast-smoke.*`: completed launcher checks.
- `completion-audit.json`: final counts, checks, file hashes and cleanup receipt.

The report, scripts and evidence live in an ignored local build directory. Git tracked files remain clean. No commit, push or external publication was made.
