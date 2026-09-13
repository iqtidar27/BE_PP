# Extension privacy-policy deduplication

Collapse Chrome Web Store policy *files* onto unique *documents* before Stage A / Stage B.

Identity = normalized text, not filename and not URL.

1. Global exact SHA-256 (`doc_id`).
2. Same-developer MinHash/LSH pairs.
3. Cross-developer MinHash/LSH pairs.
4. Union-find at estimated Jaccard ≥ **0.92** (validated on a stratified pair sample).

Outputs stay in `--out-dir` and are gitignored.

## Layout

```
policy_dedup/
  common.py
  01_dedup_same_dev.py
  02_crossdev_pairs.py
  03_export_unique.py
  review_pairs.py          # optional, threshold validation
  requirements.txt
  .gitignore
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# 1. extract, exact hash, same-developer pairs  (~50 min on ~90k files)
python3 01_dedup_same_dev.py \
  --policies-dir /path/to/selected_policies_chrome \
  --extensions-json /path/to/extensions.json \
  --out-dir /path/to/dedup_output

# 2. cross-developer pairs  (~40 min; hashes one file per doc_id)
python3 02_crossdev_pairs.py \
  --out-dir /path/to/dedup_output

# 3. unique document list at J >= 0.92
python3 03_export_unique.py \
  --out-dir /path/to/dedup_output \
  --threshold 0.92
```

Optional cutoff check (does not change clusters):

```bash
python3 review_pairs.py \
  --pairs /path/to/dedup_output/near_dup_pair_scores.csv \
  --policies-dir /path/to/selected_policies_chrome \
  --out-dir /path/to/dedup_output \
  --pack-name review_pack_samedev

python3 review_pairs.py \
  --pairs /path/to/dedup_output/crossdev_pair_scores.csv \
  --policies-dir /path/to/selected_policies_chrome \
  --out-dir /path/to/dedup_output \
  --pack-name review_pack_crossdev \
  --prefix xd_
```

Smoke test either step with `--limit 500`.

## Stage A input

`dedup_output/unique_files_j0p92.txt`  
`dedup_output/unique_files_j0p92_paths.txt`  
`dedup_output/unique_files_j0p92.csv`

## Reference numbers (CWS snapshot used to lock 0.92)

| Stage | N |
|---|---|
| Policy files | 90,813 |
| Usable | 90,780 |
| Exact unique texts | 78,313 |
| Unique after J ≥ 0.92 (same + cross dev) | **73,015** |
| Singletons | 72,108 (98.76%) |
| Multi-document families | 907 |

Do not commit `dedup_output/` or the policy file tree.
