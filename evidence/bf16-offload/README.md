# Nanbeige4.3-33B BF16 RAM-offload benchmark

Updated 2026-10-26T04:26:02.187347+00:00. Original non-base model, revision `c1093c501cc6460427c0ac634db7223d0450c0ee`. All publisher weights retained at BF16; no quantization.

Hardware: RTX5090 32GB, Ryzen9950X3D, about128GBRAM, PCIe5x16. Runtime: existing vLLM0.28.0, PyTorch2.13.0+cu132, Transformers5.16.1. Exact environment per mode is recorded in `*-environment.json`.

One request at a time, 4096context, 1GiB BF16KV cache, prefix caching disabled, eager execution. Warmup excluded. Three512-input/64-output measurements and one2048-input/64-output measurement per mode. Prompts use fixed token IDs, greedy generation, ignoreEOS and exact64-output length.

| Mode | Input tokens | Runs | Prompt tok/s | Decode tok/s | First token seconds |
|---|---:|---:|---:|---:|---:|
| UVA / direct host access | 512 | 3 | 31.08 | 0.728 | 16.478 |
| UVA / direct host access | 2048 | 1 | 36.72 | 0.718 | 55.778 |
| Async prefetch / loading workaround | 512 | 3 | 610.05 | 1.309 | 0.842 |
| Async prefetch / loading workaround | 2048 | 1 | 1762.24 | 1.315 | 1.165 |

Prompt rate uses the server's per-request prefill-time metric. Decode rate is `(output_tokens-1)/(last_text_chunk-first_text_chunk)` from localhost streaming, excluding prefill and the first token. Raw stream events, final API token counts, metric deltas, generated text and resource samples are retained.

Thinking was disabled in the chat smoke and the fixed ChatML benchmark prompts. The arithmetic chat check returned4 in both modes. All four measured64-token generated texts matched exactly across modes; this is not a full logit-parity or quality evaluation. The first512-tokenUVA request took33.17seconds to its first token, while the next two took16.47/16.48seconds; all are retained and the table uses their median.

## Memory and transfer observations

UVA logged38.23GiBhostweights and23.61GiBGPUmodel allocation. Prefetch logged40.1803GB(37.42GiB)offloaded weights and a1.1818GBstatic GPU buffer pool, with34offloaded layers. These are nearby rather than identical placements.

| Mode | Peak device VRAM, including desktop (GiB) | Benchmark process swap (bytes) |
|---|---:|---:|
| uva | 27.79 | 0 |
| prefetch | 29.58 | 0 |

Both measured servers were limited to80GiBprocess-group memory and zero process-group swap; their RAM accounting also includes checkpoint file cache, so it is not a pure weight-memory measurement. Device-widePCIe receive medians and global swap counters are in summary.json. The desktop and other host workloads were not stopped; background global swap-ins occurred, while the benchmark groups themselves used no swap and there were no global swap-outs during measured windows.

## Loading failures and workaround

Stock prefetch failed during initialization at80GiB and90GiB cgroup RAM limits, after reading weights and before serving. Source inspection found redundant GPU round-trips around the unquantized linear post-load no-op, while prefetch retained pinned host copies. The measured prefetch uses `low_peak_worker.py`, a benchmark-local startup-only workaround guarded against the exact installed no-op implementation. It changes no weights, precision, inference kernels or streaming schedule. StockUVA is the reference for paired greedy output checks. Installed vLLM and Strata source files were not edited.

The firstUVA launch loaded successfully but stopped during FlashInfer sampling JIT because the transient service PATH omitted its existingNinja executable. Corrected PATH and selected existingGCC14 for the localCUDA12.9 compiler; no dependency install. Failed logs are kept separately.

## Limits

This is a short-context, single-stream bandwidth benchmark. It does not measure512Kcontext, concurrent serving, model quality, tool-call correctness or a quantized allGPU baseline. Prefetch andUVA use nearby rather than identical host-weight budgets; exact commands and logged offload amounts are retained. Eager execution was used in both modes; this is not a claim about their highest achievable speed.

## Reproduction

The models remain in the defaultHuggingFacecache. Use the existing`~/.venvs/vllm`environment. `run_server.py uva`starts the stockUVAconfiguration; `run_server.py prefetch`starts the guarded low-peak worker. After readiness, `bench_client.py MODE`runs the fixed cases fromcases.json. Run servers one at a time, use localhostport18733, retain startup logs, and stop each server after its run. Exact commands and environment overrides are recorded in`uva-command.json`and`prefetch-command.json`; the client records runtime versions and hardware.

Paired64-token greedy text matches: {"pp512-tg64-2": true, "pp2048-tg64-0": true, "pp512-tg64-1": true, "pp512-tg64-0": true}.

Current uva state: complete.

Current prefetch state: complete.


## Decode bandwidth explanation

Decode-only device-wide PCIe receive median was 59.32 decimal GB/s across 173 samples, excluding the first-token interval plus two seconds and each final second. With 40.1803 GB of dense host weights used per decode step, transfer alone at that observed bandwidth costs about 0.677 seconds per token, equivalent to 1.48 tokens/s before compute and other overhead. The measured 512-input generation median was 1.309 tokens/s. These measurements point to PCIe weight traffic as the main bottleneck. This calculation concerns this placement and single-stream schedule; it is not a universal bound for batching, speculative decoding or other weight-reuse strategies. Raw calculation inputs and window definition are in `bandwidth-analysis.json`.
