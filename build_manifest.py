# build_manifest.py
import os, json, csv

RAW_META_DIR = "data/processed"
OUT_CSV     = "data/manifests/all_runs.csv"
HOLDOUT_SUB = "sub-04"   # change as needed

# Gather entries
entries = []
for fname in os.listdir(RAW_META_DIR):
    if not fname.endswith(".json"):
        continue
    path = os.path.join(RAW_META_DIR, fname)
    meta = json.load(open(path))
    key     = meta["key"]
    subject = key.split("_")[0]          # e.g. 'sub-01'
    split   = "test" if subject==HOLDOUT_SUB else "train"
    entries.append({"key": key,
                    "subject": subject,
                    "shape":   ";".join(map(str, meta["shape"])),
                    "split":   split})

# Write CSV
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["key","subject","shape","split"])
    writer.writeheader()
    writer.writerows(entries)

print(f"Wrote {len(entries)} entries to {OUT_CSV}")
