import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
plan = json.loads((ROOT / "capacity-plan.json").read_text())
results = []
for row in plan["estimates"]:
    if row["target_bits"] not in [0, 8, 4, 2]:
        continue
    capacity = row["predicted_capacity_with_1GiB_free"]
    for attempt in range(5):
        label = f"probe-b{row['target_bits']}-n{capacity}"
        command = [sys.executable, "-u", str(ROOT / "probe.py"), "--capacity", str(capacity),
                   "--target-bits", str(row["target_bits"]), "--draft-bits", str(row["draft_bits"]),
                   "--label", label]
        print("START", label, flush=True)
        with (ROOT / (label + ".log")).open("w") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        result = json.loads((ROOT / (label + ".json")).read_text())
        result["return_code"] = process.returncode
        result["command"] = command
        results.append(result)
        (ROOT / "allocation-results.json").write_text(json.dumps(results, indent=2) + "\n")
        if result["state"] == "passed":
            free = result["memory_after_smoke"]["free_bytes"]
            print("PASS", label, "freeGiB", free / 2**30, flush=True)
            if free >= 2**30:
                break
            reduction = math.ceil((2**30 - free + 128 * 2**20) /
                                  (row["cache_bytes_per_token"] * 256)) * 256
            capacity -= reduction
        else:
            if "out of memory" not in result.get("error", "").lower():
                raise RuntimeError(result)
            capacity -= 4096
    else:
        raise RuntimeError(f"No stable allocation for {row}")
selected = {}
for result in results:
    if result["state"] == "passed" and result["memory_after_smoke"]["free_bytes"] >= 2**30:
        selected[result["settings"]["target_bits"]] = result
(ROOT / "allocation-winners.json").write_text(json.dumps(selected, indent=2) + "\n")
print("COMPLETE", {k: v["settings"]["capacity"] for k, v in selected.items()}, flush=True)
