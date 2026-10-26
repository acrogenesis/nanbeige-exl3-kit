# Nanbeige EXL3 with DFlash2 in both current presets

Kit: https://github.com/acrogenesis/nanbeige-exl3-kit

Run the credited Nanbeige4.3-33B EXL3 3.55bpw quantization on a single RTX 5090 with DFlash2 enabled in both presets: fast generation or maximum usable context with drafting. This repository contains our launchers, benchmark harness and measured evidence. **We did not train a model, produce this quantization or create an inference engine.**

- Original model: [Nanbeige/Nanbeige4.3-33B](https://huggingface.co/Nanbeige/Nanbeige4.3-33B).
- Quantization: [ProCreations/Nanbeige4.3-33B-EXL3](https://huggingface.co/ProCreations/Nanbeige4.3-33B-EXL3/tree/56a23706262458b23ba7dde5ba1eea053a109544).
- Byte-identical model mirror: [acrogenesis/Nanbeige4.3-33B-EXL3-3.55bpw](https://huggingface.co/acrogenesis/Nanbeige4.3-33B-EXL3-3.55bpw).
- Drafter for both presets: [ProCreations/Nanbeige4.3-33B-DFlash2](https://huggingface.co/ProCreations/Nanbeige4.3-33B-DFlash2).
- Engine: stock [ExLlamaV3 1.6.5](https://github.com/turboderp-org/exllamav3/tree/v1.6.5) ([tested source commit](https://github.com/turboderp-org/exllamav3/tree/cdc785db89e263e65e5849cb53e1ada156411414)). The engine is separate from this kit and from Strata.

## Measured stats

RTX 5090 32 GB, Ryzen 9 9950X3D, about 128 GB RAM, running desktop. All rows use the same 3.55bpw weights. Measurements are from 2026-10-26.

| Test | Actual input / output | Configured capacity | KV / drafting | Output tok/s | Native prefill tok/s | TTFT |
|---|---:|---:|---|---:|---:|---:|
| Fast, median of 3 | 2,842 / 2,048 | 8,192 | FP16 / DFlash2 sparse16K | **116.92** | 3,130.03 | 926 ms |
| Max with DFlash2, median of 3 | 2,842 / 2,048 | 180,224 | Q4 target/draft / DFlash2 sparse16K | **115.14** | 3,239.42 | 896 ms |
| Near-full with DFlash2, one request | 179,705 / 256 | 180,224 | Q4 target/draft / DFlash2 sparse16K | **23.41** | 1,073.82 | 167.40 s |
| Historical maximum without drafting, median of 3 | 2,842 / 2,048 | 480,256 | Q2 / off | **70.42** | 3,156.91 | 917 ms |
| Historical near-full without drafting, one request | 479,744 / 256 | 480,256 | Q2 / off | **23.20** | 478.90 | 1,001.82 s |

The 2,842/2,048 tests used one full-length excluded warmup per preset, temperature 0, thinking off, fresh page tables and zero prefix reuse. EOS was suppressed until the fixed output length. They occupied 4,890 tokens; context capacity is a separate quantity. Output speed excludes the first streamed chunk. Prefill uses the native timer; conservative input/TTFT rates are also retained in the evidence.

The active maximum-context preset keeps DFlash2 enabled at 180,224 tokens with Q4 target/draft KV. A larger 196,608-token Q4 trial passed all three exact retrieval checks but ran out of VRAM during the separate 2,048-output-token fixed workload. We reduced the preset to leave more room for decoding buffers. This is the validated preset selected here, not a claim of the card's absolute capacity limit.

The current near-full run occupied 179,961 tokens and passed 3/3 exact retrieval checks, with 1,017 MiB CUDA VRAM free at the final snapshot. Native stats recorded 13 accepted / 1,688 rejected draft tokens: DFlash2 was active but contributed few accepted tokens at this length. Both the long and short fixed workloads fit the current preset.

A larger 344,064-token DFlash2 trial with Q2 target/draft KV executed successfully but retrieved only two of three exact values (one missing digit). It measured 15.00 output tok/s after 525.32 s to first token. We retained both larger trials and their failed checks in [the new evidence](https://github.com/acrogenesis/nanbeige-exl3-kit/blob/main/evidence/dflash-context/README.md). The single Q2/Q4 comparison does not isolate whether KV precision, context length or sampling caused the retrieval error, and neither test is a broad quality qualification.

The historical no-draft near-full test actually occupied 480,000 tokens and retrieved all three exact key/value pairs near 2%, 50% and 94% of its prompt. It used temperature 0.6/top-p 0.95. It is one synthetic retrieval test, not broad quality qualification of 2-bit KV. Full 524,288 was not executed locally. VRAM headroom at that request's final snapshot was only 157 MiB. Those two no-draft results are retained as measured history; neither describes the current `run-max.sh` preset.

An earlier five-prompt, three-seed suite measured **103.40 median / 101.77 pooled output tok/s** with 89–103 input tokens and 384 output tokens, 8K FP16 KV, DFlash2 sparse16K and temperature 0.6. Do not transfer that aggregate to all prompts or context lengths. Original BF16 RAM offload measured 0.728 tok/s (UVA) and 1.309 tok/s (prefetch) at 512 input / 64 output; the report records placement and startup differences. Those poor results remain available.

## Published results

[LocalMaxxing report and attached results](https://www.localmaxxing.com/en/models/acrogenesis/Nanbeige4.3-33B-EXL3-3.55bpw/reports/cmvpehkhi0264ng01auzavjoo) contain five approved speed entries, including the two current DFlash2 follow-ups with the exact precision, settings and raw evidence linked. [PUBLICATION.json](https://github.com/acrogenesis/nanbeige-exl3-kit/blob/main/PUBLICATION.json) records the model commit, engine revision, run IDs and readback verification; publication/ retains the submitted non-secret payloads.

## Dependency pins

The tested Transformers4.57.6 pin currently has five GitHub advisories (four high, one moderate). [DEPENDENCIES.md](https://github.com/acrogenesis/nanbeige-exl3-kit/blob/main/DEPENDENCIES.md) links them and records the untested upgrade boundary.

## Install and run

The supplied wheel was tested on NVIDIA Linux x86_64 with Python 3.12, CUDA 13.2 / torch 2.13.0 and a compatible installed NVIDIA driver. Other platforms were not tested by this kit.

```sh
git clone https://github.com/acrogenesis/nanbeige-exl3-kit
cd nanbeige-exl3-kit
./setup.sh
./run-fast.sh "Explain how database write-ahead logging supports recovery."
./run-max.sh --prompt-file document.txt --max-new-tokens 256
```

Both presets resolve the original ProCreations target and DFlash2 drafter into the default Hugging Face cache, with revisions pinned in [models.lock.json](https://github.com/acrogenesis/nanbeige-exl3-kit/blob/main/models.lock.json). Existing upstream downloads are reused. To use existing model files, pass `--target-dir /path/to/exl3`. To use an existing compatible environment, set `NANBEIGE_PYTHON=/path/to/python` when invoking a launcher.

Each invocation loads the model, streams one completion and exits. There is no background server. Default sampling is temperature 0.6/top-p 0.95, thinking off. Optional flags include `--thinking`, `--temperature`, `--max-new-tokens`, `--prompt-file`, `--output-json` and `--cache-tokens`.

Context includes input, reserved output and 256 tokens for bookkeeping, plus seven draft proposals. Oversized requests are rejected before GPU model loading. The fast preset's cache is 8,192 tokens with FP16 target/draft KV; the max preset is 180,224 with 4-bit target/draft KV. Both use DFlash2 with seven proposals and a 16K sparse draft vocabulary. Reduce `--cache-tokens` in multiples of 256 if more VRAM headroom is needed. Max sets `EXL3_QC_STAGING_PAGES=64`, a 16K compressed-KV staging buffer that bounds scratch while still attending to the older slices.

Both launchers seed the stock engine's sparse vocabulary frequency profile only if one is absent. The seed used in our measurements is retained in profiles/. Benchmark runs use an isolated private copy and do not overwrite the user's learned profile.

## Reproduce the benchmarks

```sh
# One full warmup, then three measured 2,842-input / 2,048-output requests.
.venv/bin/python benchmark.py --preset fast
.venv/bin/python benchmark.py --preset max

# Actually fill nearly the whole maximum-context cache.
EXL3_QC_STAGING_PAGES=64 .venv/bin/python benchmark_full.py \
  --capacity 180224 --target-bits 4 --draft-bits 4 --label full180k-b4-dflash \
  --output-tokens 256 --temperature 0.6

# Earlier five-prompt natural-completion suite.
.venv/bin/python bench_exl3.py --mode draft --label natural-suite \
  --kv-bits 0 --sparse-draft-head 16384 --no-thinking
```

Run GPU profiles sequentially. New fixed-workload/full-context results go under ignored runs/. The natural suite retains its original current-directory output convention. Long prefills can take several minutes on the measured hardware. Archived original sources and raw measurements are under evidence/; root-level scripts are the portable interfaces. Historical no-draft runs can still be reproduced with the full-context harness's explicit `--no-draft` flag. The earlier fixed-workload interfaces are preserved at kit commit `41888b4`; current `benchmark.py --preset max` always uses DFlash2.

## Evidence and validation

- [Stats card workload](https://github.com/acrogenesis/nanbeige-exl3-kit/blob/main/evidence/card-stats/README.md): per-token arrival times, all generated output, counts, medians, ranges, telemetry and independent audit.
- [Historical near-full context without drafting](https://github.com/acrogenesis/nanbeige-exl3-kit/blob/main/evidence/max-context/README.md): exact 479,744 input IDs, saved archive, three exact retrieval targets, all output, memory and timing audit.
- [DFlash2 maximum context](https://github.com/acrogenesis/nanbeige-exl3-kit/blob/main/evidence/dflash-context/README.md): current preset, allocation checks, fixed workload and actual near-full retrieval with drafting enabled.
- [Five-prompt short suite](https://github.com/acrogenesis/nanbeige-exl3-kit/blob/main/evidence/short-suite/README.md): original speed tuning and broader prompt results.
- [BF16 RAM offload](https://github.com/acrogenesis/nanbeige-exl3-kit/blob/main/evidence/bf16-offload/README.md): original full-precision baseline and measured PCIe bottleneck.

The public kit's launchers are checked with an actual `17 * 23` completion and native EOS. The portable fixed-length and archive benchmark paths are also exercised; their validation receipt is published with the evidence. The original nearly 480K GPU test was not rerun solely for moving path resolution into this package. Setup dependencies were exercised in the existing isolated environment; a fresh installation from setup.sh is not claimed to have been tested.

Public evidence replaces the machine's home path with `${HOME}`. Raw token IDs, generated text and measured rates are retained. Archived source files may reference the original local directory layout; they are historical evidence, not a new engine. No environment files or API keys are included.

The model mirror retains Apache-2.0 LICENSE/NOTICE and upstream documentation. Kit scripts and our benchmark material use the MIT license below; upstream engine/model/drafter licenses remain separate. No claims of BF16 parity, a new quality evaluation, native Strata compatibility or universal throughput are made.
