# Local measurements

Single RTX 5090, single request at a time. Nanbeige4.3-33B at the stated EXL3 precision. Target KV is Q8 unless the configuration is labeled f16kv; draft KV is FP16. Thinking tokens count when thinking is enabled. Prompt processing and warmup are excluded from decode. Fresh page tables for every request; zero cached prompt tokens verified.

Decode rate is (tokens emitted after the first chunk) / (last arrival - first arrival). This excludes the entire first speculative chunk. The per-run end-to-end rate also includes prompt time. Output caps are stated in each run's settings; EOS is respected. This is a speed investigation, not a full model-quality evaluation.

| Configuration | Runs | Decode median | Decode pooled | Range | End-to-end median | First-token median |
|---|---:|---:|---:|---:|---:|---:|
| plain-4.05-t06-thinking | 5 | 69.57 | 69.61 | 69.54–69.72 | 68.86 | 0.070 s |
| draft7-4.05-t06-thinking | 15 | 77.94 | 81.75 | 65.64–111.25 | 76.98 | 0.074 s |
| draft7-4.05-t06-nothinking | 15 | 89.61 | 94.86 | 78.22–121.06 | 87.84 | 0.074 s |
| draft3-4.05-t06-nothinking | 5 | 97.47 | 98.48 | 86.89–114.60 | 95.91 | 0.074 s |
| plain-4.05-t06-thinking-f16kv | 3 | 70.43 | 70.42 | 70.30–70.53 | 69.74 | 0.068 s |
| draft7-3.55-t06-nothinking-f16kv | 15 | 95.88 | 98.26 | 85.13–135.69 | 94.42 | 0.072 s |
| draft7-3.55-t0-nothinking-f16kv | 15 | 92.79 | 98.44 | 84.45–130.80 | 90.92 | 0.072 s |
| draft7-3.55-t06-nothinking-f16kv-v165-sparse16k-device | 15 | 103.40 | 101.77 | 74.12–136.91 | 101.06 | 0.072 s |

## plain-4.05-t06-thinking

Prompt-type medians (tokens/s):

- code: 69.72
- debugging: 69.57
- math: 69.54
- explanation: 69.56
- chinese: 69.68

## draft7-4.05-t06-thinking

Prompt-type medians (tokens/s):

- code: 87.79
- debugging: 76.36
- math: 106.88
- explanation: 77.16
- chinese: 66.25

## draft7-4.05-t06-nothinking

Prompt-type medians (tokens/s):

- code: 110.73
- debugging: 85.81
- math: 110.73
- explanation: 86.87
- chinese: 84.50

## draft3-4.05-t06-nothinking

Prompt-type medians (tokens/s):

- code: 114.60
- debugging: 89.51
- math: 109.83
- explanation: 86.89
- chinese: 97.47

## plain-4.05-t06-thinking-f16kv

Prompt-type medians (tokens/s):

- code: 70.53
- math: 70.43
- chinese: 70.30

## draft7-3.55-t06-nothinking-f16kv

Prompt-type medians (tokens/s):

- code: 121.45
- debugging: 90.79
- math: 105.25
- explanation: 88.84
- chinese: 92.56

## draft7-3.55-t0-nothinking-f16kv

Prompt-type medians (tokens/s):

- code: 130.44
- debugging: 92.79
- math: 110.39
- explanation: 87.21
- chinese: 84.67

## draft7-3.55-t06-nothinking-f16kv-v165-sparse16k-device

Prompt-type medians (tokens/s):

- code: 131.49
- debugging: 94.96
- math: 112.69
- explanation: 91.38
- chinese: 98.67

Raw per-request token IDs, text, event times, draft acceptance records, settings and runtime versions are in the corresponding JSON files. Runtime installation log, exact model revisions, remote file metadata, full weight integrity checks and GPU telemetry are retained alongside them.
