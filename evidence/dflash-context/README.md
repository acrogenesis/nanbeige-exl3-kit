# DFlash2 retained in both active presets

Measured 2026-10-26 on RTX 5090 32GB, Ryzen 9950X3D,128GB RAM, running desktop. Same ProCreations3.55bpw target, pinned native DFlash2 and stock ExLlamaV3 1.6.5. Both active launchers and fixed-workload presets use DFlash2 with seven proposals and sparse16K vocabulary. Original target/draft downloads in the standard Hugging Face cache are reused.

| Trial | Capacity | Actual input/output | Target/draft KV | Output tok/s | Prefill tok/s | TTFT | Outcome |
|---|---:|---:|---|---:|---:|---:|---|
| Current fixed workload, median3 | 180224 | 2842/2048 | Q4/Q4 | 115.135906 | 3239.416026 | 895.616ms | 3/3 requests passed |
| Current near-full, one request | 180224 | 179705/256 | Q4/Q4 | 23.405918 | 1073.817987 | 167.400490s | 3/3 exact bindings |
| Larger Q4 near-full | 196608 | 196089/256 | Q4/Q4 | 22.144024 | 1009.473649 | 194.300643s | Exact retrieval passed; separate2048-output warmup OOM |
| Larger Q2 near-full | 344064 | 343545/256 | Q2/Q2 | 15.004597 | 654.062266 | 525.322693s | 2/3 exact bindings; cedar lost final digit |

The selected180224 capacity passed both workloads and left1017.125MiB CUDA free at the near-full final snapshot. This is a validated practical preset under the measured desktop load, not an exhaustive maximum-capacity proof. Larger196608 Q4 passed its full archive test but failed the separate greedy fixed-output warmup with CUDA out of memory, so it is not the active preset. Larger344064 Q2 executed but its exact retrieval failed. Failures and all larger-trial prompts/output/audits are retained. The Q2/Q4 comparison changes context and sampling path, so it does not isolate the cause of the missing digit.

Fixed workload: exactly2842input/2048output; one full excluded warmup+three measured requests; T0, thinking off, EOS suppressed until2048, fresh generator/page tables per request, zero cached prefix tokens. Decode excludes the first streamed chunk; native prefill and enqueue-to-first-chunk TTFT are separate. Isolated sparse profile starts from the published seed and learns during warmup/requests. The short-workload output range was 115.047859–115.256265tok/s.

Near-full: T0.6/top_p0.95, natural EOS or256-token cap, excluded64-token warmup, fresh page tables, zero prefix reuse. Markers at approximately2%,50%,94%; exact JSON key/value checks, prompt hashes, raw token counts and per-token timing recomputation are audited. There was one synthetic long-context request per configuration, not a broad quality evaluation or BF16 parity check. Long-context draft acceptance is low despite DFlash2 being enabled: current near-full accepted13/proposed1701. No guarantee that drafting speeds up every workload.

Original allocation rechecks are retained: FP16 draft-cache variants372224/335872 OOMed; Q2/Q2 at344064 passed short allocation smokes before its later retrieval failure. These are different cache precisions and must not be conflated. Telemetry is1Hz and includes the desktop; resource-summary.json filters the successful request windows.

From the kit root:

```sh
.venv/bin/python benchmark.py --preset max
EXL3_QC_STAGING_PAGES=64 .venv/bin/python benchmark_full.py --capacity 180224 --target-bits 4 --draft-bits 4 --label full180k-b4-dflash --output-tokens 256 --temperature 0.6
python evidence/dflash-context/audit.py
```

The independent audit also checks that the larger Q2 trial remains an expected failure and that the larger Q4 fixed-workload OOM is retained. Launcher smoke returns391 for17*23 and native EOS. Engine source is unchanged. Archived480K/no-draft results remain historical.
