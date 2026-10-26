"""Recompute the stats card metrics from raw token arrivals and native metadata."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
ids = json.loads((ROOT / "prompt-ids.json").read_text())
digest = hashlib.sha256((ROOT / "prompt.txt").read_bytes()).hexdigest()
assert len(ids) == 2842
result = {"state": "passed", "input_tokens": 2842, "output_tokens": 2048,
          "occupied_tokens": 4890, "prompt_sha256": digest, "profiles": {}}
for preset in ["fast", "max"]:
    path = ROOT / (preset + ".json")
    report = json.loads(path.read_text())
    assert report["state"] == "complete", (preset, report["state"])
    assert report["prompt_sha256"] == digest
    assert len(report["results"]) == 4
    rows = [r for r in report["results"] if r["measured"]]
    assert len(rows) == 3 and not report["results"][0]["measured"]
    for row in report["results"]:
        terminal = row["terminal"]
        assert row["input_tokens"] == terminal["prompt_tokens"] == len(ids)
        assert row["output_tokens"] == terminal["new_tokens"] == 2048
        assert row["occupied_tokens"] == 4890
        assert terminal["cached_tokens"] == 0
        assert terminal["eos"] and terminal["eos_reason"] == "max_new_tokens"
        events = row["events"]
        assert [t for e in events for t in e["token_ids"]] == row["output_token_ids"]
        assert len(row["output_token_ids"]) == 2048
        assert all(a["seconds"] <= b["seconds"] for a, b in zip(events, events[1:]))
        output_rate = (2048 - len(events[0]["token_ids"])) / (events[-1]["seconds"] - events[0]["seconds"])
        ttft_ms = events[0]["seconds"] * 1000
        prefill_rate = 2842 / terminal["time_prefill"]
        for value, key in [(output_rate, "output_tokens_s"), (ttft_ms, "ttft_ms"),
                           (prefill_rate, "prefill_tokens_s")]:
            assert abs(row[key] - value) < 1e-8
    medians = {k: statistics.median(row[k] for row in rows) for k in report["medians"]}
    assert medians == report["medians"]
    result["profiles"][preset] = {"capacity": report["capacity"],
        "weight_bpw": report["weight_bpw"], "kv_bits": report["kv_bits"],
        "draft": report["draft"], "measured_requests": len(rows),
        "medians": medians, "output_rate_range": report["output_rate_range"],
        "all_measured_outputs_identical": len({tuple(r["output_token_ids"]) for r in rows}) == 1}
(ROOT / "stats-audit.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
