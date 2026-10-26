import argparse
import hashlib
import json
from pathlib import Path
import re

p = argparse.ArgumentParser()
p.add_argument("result")
a = p.parse_args()
path = Path(a.result)
r = json.loads(path.read_text())
assert r["state"] == "complete", r["state"]
prompt = path.with_name(path.stem + "-prompt.txt").read_text()
prompt_ids = json.loads(path.with_name(path.stem + "-prompt-ids.json").read_text())
assert len(prompt_ids) == r["settings"]["input_tokens"]
assert hashlib.sha256(prompt.encode()).hexdigest() == r["prompt_sha256"]
expected = {m["key"]: m["value"] for m in r["markers"]}
for value in expected.values():
    assert prompt.count(value) == 1
    assert value not in prompt.rsplit("</archive>", 1)[1]
mapping = None
for text in re.findall(r"\{[^{}]*\}", r["text"], re.DOTALL):
    try:
        candidate = json.loads(text)
    except ValueError:
        continue
    if isinstance(candidate, dict) and all(k in candidate for k in expected):
        mapping = candidate
        break
checks = {k: mapping is not None and mapping.get(k) == v for k, v in expected.items()}
assert r["terminal"]["cached_tokens"] == 0
assert r["terminal"]["prompt_tokens"] == r["settings"]["input_tokens"]
assert sum(len(e["token_ids"]) for e in r["events"]) == r["output_tokens"]
assert [t for e in r["events"] for t in e["token_ids"]] == r["output_token_ids"]
assert r["terminal"]["new_tokens"] == r["output_tokens"]
assert r["text"] == r["terminal"]["full_completion"]
assert r["occupied_context_tokens"] == len(prompt_ids) + r["output_tokens"]
assert r["occupied_context_tokens"] + 256 <= r["settings"]["capacity"]
count = r["output_tokens"] - len(r["events"][0]["token_ids"])
elapsed = r["events"][-1]["seconds"] - r["events"][0]["seconds"]
assert abs(count / elapsed - r["decode_tokens_s"]) < 1e-9
result = {"result_file": path.name, "state": "passed" if all(checks.values()) else "failed",
          "expected": expected, "received": mapping, "exact_key_value_checks": checks,
          "markers": r["markers"], "capacity": r["settings"]["capacity"],
          "input_tokens": r["settings"]["input_tokens"], "output_tokens": r["output_tokens"],
          "occupied_context_tokens": r["occupied_context_tokens"],
          "decode_tokens_s_recomputed": count / elapsed, "ttft_s": r["ttft_s"],
          "prompt_hash_verified": True, "raw_token_counts_verified": True,
          "values_unique_and_absent_from_question": True, "cached_prompt_tokens": 0,
          "limits": "One synthetic archive retrieval test, three key/value pairs; not a comprehensive long-context quality evaluation."}
output = path.with_name(path.stem + "-audit.json")
output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
raise SystemExit(0 if all(checks.values()) else 1)
