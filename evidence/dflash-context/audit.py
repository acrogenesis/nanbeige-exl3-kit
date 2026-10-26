"""Check the new DFlash2 records against saved tokens and native metadata."""
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

root = Path(__file__).resolve().parent
fixed = json.loads((root / "max.json").read_text())
assert fixed["state"] == "complete"
assert fixed["draft"] and fixed["kv_bits"] == fixed["draft_kv_bits"] == 4
assert fixed["sparse_draft_head"] == 16384
prompt_ids = json.loads((root / "prompt-ids.json").read_text())
assert len(prompt_ids) == 2842
assert hashlib.sha256((root / "prompt.txt").read_bytes()).hexdigest() == fixed["prompt_sha256"]
assert len(fixed["results"]) == 4 and not fixed["results"][0]["measured"]
for row in fixed["results"]:
    terminal = row["terminal"]
    assert row["input_tokens"] == terminal["prompt_tokens"] == 2842
    assert row["output_tokens"] == terminal["new_tokens"] == 2048
    assert row["occupied_tokens"] == 4890
    assert terminal["cached_tokens"] == 0
    assert terminal["eos_reason"] == "max_new_tokens"
    assert terminal["accepted_draft_tokens"] > 0
    events = row["events"]
    assert [token for event in events for token in event["token_ids"]] == row["output_token_ids"]
    assert len(row["output_token_ids"]) == 2048
    assert all(x["seconds"] <= y["seconds"] for x, y in zip(events, events[1:]))
    recomputed = {
        "output_tokens_s": (2048 - len(events[0]["token_ids"])) / (events[-1]["seconds"] - events[0]["seconds"]),
        "ttft_ms": events[0]["seconds"] * 1000,
        "prefill_tokens_s": 2842 / terminal["time_prefill"],
    }
    for key, value in recomputed.items():
        assert abs(row[key] - value) < 1e-8
measured = [row for row in fixed["results"] if row["measured"]]
assert len(measured) == 3
assert fixed["medians"] == {key: statistics.median(row[key] for row in measured) for key in fixed["medians"]}

full_path = root / "full180k-b4-dflash.json"
subprocess.run([sys.executable, str(root.parent / "max-context/audit_retrieval.py"), str(full_path)], check=True)
full = json.loads(full_path.read_text())
assert full["draft_dir"] and not full["settings"]["no_draft"]
assert full["settings"]["target_bits"] == full["settings"]["draft_bits"] == 4
assert full["settings"]["capacity"] == fixed["capacity"]
assert full["terminal"]["accepted_draft_tokens"] + full["terminal"]["rejected_draft_tokens"] > 0
assert full["occupied_context_tokens"] + 256 + 7 <= full["settings"]["capacity"]
summary = {
    "state": "passed",
    "capacity": fixed["capacity"],
    "fixed_workload": {"actual_input": 2842, "actual_output": 2048, "measured_requests": 3,
                       "medians": fixed["medians"], "range": fixed["output_rate_range"]},
    "full_workload": {"actual_input": full["settings"]["input_tokens"], "actual_output": full["output_tokens"],
                      "occupied_tokens": full["occupied_context_tokens"], "output_tokens_s": full["decode_tokens_s"],
                      "prefill_tokens_s": full["settings"]["input_tokens"] / full["terminal"]["time_prefill"],
                      "ttft_ms": full["ttft_s"] * 1000,
                      "final_cuda_free_MiB": full["memory_final"]["free_bytes"] / 2**20},
    "draft_enabled_in_all_requests": True,
    "native_draft_activity_verified": True,
    "counts_timings_prompt_hashes_and_exact_retrieval_verified": True,
    "limits": "One synthetic long-context retrieval request per configuration; no broad KV quality evaluation or BF16 parity claim.",
}
failed_path = root / "full344k-b2-dflash.json"
failed_check = subprocess.run([sys.executable, str(root.parent / "max-context/audit_retrieval.py"), str(failed_path)])
assert failed_check.returncode == 1
failed_audit = json.loads(failed_path.with_name(failed_path.stem + "-audit.json").read_text())
assert failed_audit["state"] == "failed"
assert sum(failed_audit["exact_key_value_checks"].values()) == 2
summary["larger_2bit_trial"] = {"capacity": 344064, "exact_retrieval": "2/3", "usable_preset": False,
                               "result": failed_path.name, "audit": failed_path.stem + "-audit.json"}
oom = json.loads((root / "max196k-b4-oom.json").read_text())
assert oom["state"] == "failed" and "out of memory" in oom["error"].lower()
larger_audit = json.loads((root / "full196k-b4-dflash-audit.json").read_text())
assert larger_audit["state"] == "passed"
summary["larger_4bit_trial"] = {"capacity": 196608, "exact_retrieval": "3/3",
                               "fixed_length_warmup": "out of memory", "usable_preset": False}
(root / "audit.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
