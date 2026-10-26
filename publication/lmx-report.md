# Nanbeige EXL3 inference kit

Run the credited Nanbeige4.3-33B EXL3 3.55bpw quantization on a single RTX 5090 with two tested presets: fast generation or maximum usable context. This repository contains our launchers, benchmark harness and measured evidence. **We did not train a model, produce this quantization or create an inference engine.**

- Original model: [Nanbeige/Nanbeige4.3-33B](https://huggingface.co/Nanbeige/Nanbeige4.3-33B).
- Quantization: [ProCreations/Nanbeige4.3-33B-EXL3](https://huggingface.co/ProCreations/Nanbeige4.3-33B-EXL3/tree/56a23706262458b23ba7dde5ba1eea053a109544).
- Byte-identical model mirror: [acrogenesis/Nanbeige4.3-33B-EXL3-3.55bpw](https://huggingface.co/acrogenesis/Nanbeige4.3-33B-EXL3-3.55bpw).
- Optional drafter: [ProCreations/Nanbeige4.3-33B-DFlash2](https://huggingface.co/ProCreations/Nanbeige4.3-33B-DFlash2).
- Engine: stock [ExLlamaV3 1.6.5](https://github.com/turboderp-org/exllamav3/tree/v1.6.5), with an [unchanged source fork pinned to the tested release](https://github.com/acrogenesis/exllamav3/tree/nanbeige-tested-v1.6.5). The engine is separate from this kit and from Strata.

## Measured stats

RTX 5090 32 GB, Ryzen 9 9950X3D, about 128 GB RAM, running desktop. All rows use the same 3.55bpw weights. Measurements are from 2026-10-26.

| Test | Actual input / output | Configured capacity | KV / drafting | Output tok/s | Native prefill tok/s | TTFT |
|---|---:|---:|---|---:|---:|---:|
| Fast, median of 3 | 2,842 / 2,048 | 8,192 | FP16 / DFlash2 sparse16K | **116.92** | 3,130.03 | 926 ms |
| Max preset, median of 3 | 2,842 / 2,048 | 480,256 | Q2 / off | **70.42** | 3,156.91 | 917 ms |
| Actually near-full, one request | 479,744 / 256 | 480,256 | Q2 / off | **23.20** | 478.90 | 1,001.82 s |

The 2,842/2,048 tests used one full-length excluded warmup per preset, temperature 0, thinking off, fresh page tables and zero prefix reuse. EOS was suppressed until the fixed output length. They occupied 4,890 tokens; context capacity is a separate quantity. Output speed excludes the first streamed chunk. Prefill uses the native timer; conservative input/TTFT rates are also retained in the evidence.

The near-full test actually occupied 480,000 tokens and retrieved all three exact key/value pairs near 2%, 50% and 94% of its prompt. It used temperature 0.6/top-p 0.95. It is one synthetic retrieval test, not broad quality qualification of 2-bit KV. Full 524,288 was not executed locally. VRAM headroom at the near-full request's final snapshot was only 157 MiB; other GPU applications may require a smaller cache.

An earlier five-prompt, three-seed suite measured **103.40 median / 101.77 pooled output tok/s** with 89–103 input tokens and 384 output tokens, 8K FP16 KV, DFlash2 sparse16K and temperature 0.6. Do not transfer that aggregate to all prompts or context lengths. Original BF16 RAM offload measured 0.728 tok/s (UVA) and 1.309 tok/s (prefetch) at 512 input / 64 output; the report records placement and startup differences. Those poor results remain available.

## Install and run

The supplied wheel was tested on NVIDIA Linux x86_64 with Python 3.12, CUDA 13.2 / torch 2.13.0 and a compatible installed NVIDIA driver. Other platforms were not tested by this kit.

```sh
git clone https://github.com/acrogenesis/nanbeige-exl3-kit
cd nanbeige-exl3-kit
./setup.sh
./run-fast.sh "Explain how database write-ahead logging supports recovery."
./run-max.sh --prompt-file document.txt --max-new-tokens 256
```

Models download into the default Hugging Face cache, with revisions pinned in [models.lock.json](https://github.com/acrogenesis/nanbeige-exl3-kit/blob/5ed9c48984d95f62d386055248a7000245c272c5/models.lock.json). The maximum preset does not download or load the drafter. To use existing model files, pass `--target-dir /path/to/exl3`. To use an existing compatible environment, set `NANBEIGE_PYTHON=/path/to/python` when invoking a launcher.

Each invocation loads the model, streams one completion and exits. There is no background server. Default sampling is temperature 0.6/top-p 0.95, thinking off. Optional flags include `--thinking`, `--temperature`, `--max-new-tokens`, `--prompt-file`, `--output-json` and `--cache-tokens`.

Context includes input, reserved output and 256 tokens for bookkeeping, plus draft proposals when applicable. Oversized requests are rejected before GPU model loading. The fast preset's cache is 8,192 tokens; the max preset is 480,256. Reduce `--cache-tokens` in multiples of 256 if more VRAM headroom is needed. Max sets `EXL3_QC_STAGING_PAGES=64`, a 16K compressed-KV staging buffer that bounds scratch while still attending to the older slices.

The fast launcher seeds the stock engine's sparse vocabulary frequency profile only if one is absent. The seed used in our measurements is retained in profiles/. Benchmark runs use an isolated private copy and do not overwrite the user's learned profile.

## Reproduce the benchmarks

```sh
# One full warmup, then three measured 2,842-input / 2,048-output requests.
.venv/bin/python benchmark.py --preset fast
.venv/bin/python benchmark.py --preset max

# Actually fill nearly the whole maximum-context cache.
EXL3_QC_STAGING_PAGES=64 .venv/bin/python benchmark_full.py \
  --capacity 480256 --target-bits 2 --no-draft --label full480k-b2-plain \
  --output-tokens 256 --temperature 0.6

# Earlier five-prompt natural-completion suite.
.venv/bin/python bench_exl3.py --mode draft --label natural-suite \
  --kv-bits 0 --sparse-draft-head 16384 --no-thinking
```

Run GPU profiles sequentially. New fixed-workload/full-context results go under ignored runs/. The natural suite retains its original current-directory output convention. The full prompt can take about 17 minutes to read on the measured hardware. Archived original sources and raw measurements are under evidence/; root-level scripts are the portable interfaces.

## Evidence and validation

- [Stats card workload](https://github.com/acrogenesis/nanbeige-exl3-kit/blob/5ed9c48984d95f62d386055248a7000245c272c5/evidence/card-stats/README.md): per-token arrival times, all generated output, counts, medians, ranges, telemetry and independent audit.
- [Actual near-full context](https://github.com/acrogenesis/nanbeige-exl3-kit/blob/5ed9c48984d95f62d386055248a7000245c272c5/evidence/max-context/README.md): exact 479,744 input IDs, saved archive, three exact retrieval targets, all output, memory and timing audit.
- [Five-prompt short suite](https://github.com/acrogenesis/nanbeige-exl3-kit/blob/5ed9c48984d95f62d386055248a7000245c272c5/evidence/short-suite/README.md): original speed tuning and broader prompt results.
- [BF16 RAM offload](https://github.com/acrogenesis/nanbeige-exl3-kit/blob/5ed9c48984d95f62d386055248a7000245c272c5/evidence/bf16-offload/README.md): original full-precision baseline and measured PCIe bottleneck.

The public kit's launchers are checked with an actual `17 * 23` completion and native EOS. The portable fixed-length and archive benchmark paths are also exercised; their validation receipt is published with the evidence. The original nearly 480K GPU test was not rerun solely for moving path resolution into this package. Setup dependencies were exercised in the existing isolated environment; a fresh installation from setup.sh is not claimed to have been tested.

Public evidence replaces the machine's home path with `${HOME}`. Raw token IDs, generated text and measured rates are retained. Archived source files may reference the original local directory layout; they are historical evidence, not a new engine. No environment files or API keys are included.

The model mirror retains Apache-2.0 LICENSE/NOTICE and upstream documentation. Kit scripts and our benchmark material use the MIT license below; upstream engine/model/drafter licenses remain separate. No claims of BF16 parity, a new quality evaluation, native Strata compatibility or universal throughput are made.


## Publication identity

Model weights are mirrored byte-for-byte from the credited ProCreations revision. No new model training, quantization or inference engine was created. Our contribution is the tested presets, launchers and benchmark evidence. All three attached speed entries use the same mirrored weights and stock ExLlamaV31.6.5; fixed-length and actual near-full workloads remain separate. The API key used for submission remains local and is absent from public files.
