# Nanbeige4.3-33B at 100 generated tokens/s

Measured on this RTX 5090 32 GB, Ryzen 9950X3D and 128 GB RAM: **103.40 tokens/s median, 101.77 tokens/s pooled decode**, across 15 single-request measurements. End-to-end median including prompt processing was 101.06 tokens/s. This uses Nanbeige4.3-33B EXL3 3.55bpw, DFlash2, ExLlamaV3 1.6.5 and its opt-in sparse drafting head. **Thinking is off.** Individual requests ranged from 74.12 to 136.91 tokens/s; 100 is not a per-request or all-context guarantee.

## Reproduce and use

From the Strata checkout:

```sh
./build-nanbeige-evidence/2026-10-25-100tps/run-fast.sh "Explain how a database B-tree index works."
```

The launcher streams a response and prints its measured generation rate. It loads the model for each invocation; first-use compilation and model startup add latency. `--thinking` enables reasoning but its speed with this final configuration was not measured. `--max-new-tokens` defaults to 1024, `--temperature` to 0.6. Add `--output-json /absolute/path/result.json` to retain token IDs and timing. This is a CLI, not a persistent API server.

Full results: [RESULTS.md](RESULTS.md). Frozen winning settings: [winning-config.json](winning-config.json). Independent token-count/timing recomputation: [measurement-audit.json](measurement-audit.json).

## Hardware, software and weights

RTX 5090 32 GB, stock 600 W power limit, driver 615.78.08, Ryzen 9950X3D, about 128 GB host RAM. No clock, power, desktop or system configuration changes. Existing vLLM environment and Strata source remain unchanged.

Final isolated runtime: `~/.venvs/nanbeige-exl3-165`, ExLlamaV3 1.6.5+cu132.torch2.13.0, Torch 2.13.0+cu132, Transformers 4.57.6. Exact dependencies: `runtime-165-freeze.txt`. Earlier 1.6.4 environment and results remain available. Publisher tested the native draft with ExLlamaV3 1.6.4 / Torch 2.10 / CUDA 12.8; our package build differs and was exercised locally.

All weights are in the default Hugging Face cache. Original base and non-base BF16 models remain there. Newly downloaded artifacts:

- Target: ProCreations/Nanbeige4.3-33B-EXL3, 3.55bpw, revision `56a23706262458b23ba7dde5ba1eea053a109544`.
- Draft: ProCreations/Nanbeige4.3-33B-DFlash2, native exllamav3 branch, revision `4b7c7f8509d5286b7264db096d764b4a163d74f4`.
- Earlier target: same EXL3 repository, 4.05bpw, revision `0cf884698824abbf9a9192d067cd0b3457a048f9`.

Full LFS hashes were checked: `integrity.json` and `3.55-model.json`. Remote file metadata, publisher instructions and config snapshots are retained. No local full model-quality evaluation or comparison with BF16 logits was performed. Quantization changes numerical behavior.

## Measurement

Five natural prompt types: Python code, payment-workflow design/debugging, mathematics, database explanation, and Chinese inventory-system design. Input lengths are 89–103 tokens, not an occupied 8K context. Each measured request generated 384 tokens; 3 seeds per prompt type. Temperature 0.6, top_p 0.95, thinking off, EOS respected. Five complete warmup requests excluded. Requests are serial, with cache capacity 8192 tokens and FP16 target/draft KV. CUDA graphs on, 7 draft proposals, OMP_NUM_THREADS=4. Fresh page tables for every request; zero cached prompt tokens verified.

Decode is emitted tokens after the first chunk divided by time between first and last emitted chunks. This excludes the whole first speculative chunk, not just one token. Pooled decode divides all those tokens by the sum of their decode times. Native token counts and event token counts match for all 15 requests. No end-of-message tokens occur inside any of the measured 384-token outputs.

The 16K-capacity sparse draft head learns token frequencies from prompts and emitted text. It only restricts drafting candidates; the stock engine still verifies with the full target vocabulary. The winning run began with an empty isolated frequency profile, then warmed all five prompt types. Its actual active subset grew to about 1,300 observed tokens. The resulting profile was copied to the library's default cache for the launcher; copy receipt is `runtime-profile-seed.json`. Benchmark profile stays frozen in the evidence directory. Adaptation and prompt content affect speed.

The host desktop stayed running. GPU telemetry spans all candidate runs, with explicit idle gaps. Draft-window and FP16-cache micro-comparisons overlapped the 3.55bpw download; final winning measurements started after its download and full hash check finished. This is a local short-prompt speed result, not a long-context, concurrency, agent-tool correctness or full quality qualification.

## Alternatives and failures retained

4.05bpw without drafting: 69.57 tokens/s median. With seven-proposal DFlash2: 77.94 thinking-on, 89.61 thinking-off. Shortening the draft window did not provide a clear win; FP16 KV gave about a 1% plain-decode improvement. 3.55bpw with stock 1.6.4: 95.88 median at temperature 0.6, 92.79 greedy. Faster code/math requests alone were not treated as the full target.

ExLlamaV3 1.6.5 first sparse launch failed before generating: its sparse head expected `linear.device.type`, while the publisher-style `model.load(device="cuda:0")` stored a string. Passing the documented `torch.device("cuda:0")` argument fixed it without installed source edits. The failed log and partial result remain separate. The model's publisher-style PreTrainedTokenizerFast loader emits a tokenizer-class warning; local chat-template use and actual generation succeeded.

The CLI initially displayed the held EOS marker after the correct answer; it now streams only emitted completion text while retaining all native token IDs in evidence. This presentation-only change does not alter the benchmark path or timing calculation.

Primary instructions: https://huggingface.co/ProCreations/Nanbeige4.3-33B-DFlash2/tree/exllamav3
Runtime release: https://github.com/turboderp-org/exllamav3/releases/tag/v1.6.5
