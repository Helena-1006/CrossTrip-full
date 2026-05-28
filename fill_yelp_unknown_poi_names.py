#!/usr/bin/env python3
"""Fill UNKNOWN POI names in enriched Yelp files using offline mappings.

Strategy (offline only, no API calls):
1) Build business_id -> poi_name map from yelp_business_cache.json (valid names only).
2) Merge with known names found in enriched_home.txt and enriched_oot.txt.
3) Replace row[6] == "[UNKNOWN_POI]" when business_id is in map.

Files are overwritten in place by default (optionally keep backups).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

UNKNOWN = "[UNKNOWN_POI]"


def read_tsv(path: Path) -> List[List[str]]:
    rows: List[List[str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            rows.append(line.split("\t"))
    return rows


def write_tsv(path: Path, rows: List[List[str]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write("\t".join(row) + "\n")


def load_cache_name_map(cache_path: Path) -> Dict[str, str]:
    if not cache_path.exists():
        return {}

    data = json.loads(cache_path.read_text(encoding="utf-8"))
    name_map: Dict[str, str] = {}
    for bid, obj in data.items():
        name = str((obj or {}).get("name") or "").strip()
        if name and name != UNKNOWN:
            name_map[bid] = name
    return name_map


def extract_name_map_from_rows(rows: List[List[str]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for row in rows:
        if len(row) < 8:
            continue
        bid = row[3]
        name = row[6].strip()
        if name and name != UNKNOWN:
            out[bid] = name
    return out


def count_unknown(rows: List[List[str]]) -> Tuple[int, int]:
    total = 0
    unknown = 0
    for row in rows:
        if len(row) < 8:
            continue
        total += 1
        if row[6].strip() == UNKNOWN:
            unknown += 1
    return total, unknown


def fill_unknown(rows: List[List[str]], name_map: Dict[str, str]) -> Tuple[List[List[str]], int]:
    filled = 0
    out: List[List[str]] = []
    for row in rows:
        if len(row) < 8:
            out.append(row)
            continue

        bid = row[3]
        name = row[6].strip()
        if name == UNKNOWN and bid in name_map:
            row = row[:]
            row[6] = name_map[bid]
            filled += 1
        out.append(row)
    return out, filled


def unresolved_unique_bids(rows_a: List[List[str]], rows_b: List[List[str]]) -> int:
    unresolved = set()
    for rows in (rows_a, rows_b):
        for row in rows:
            if len(row) < 8:
                continue
            if row[6].strip() == UNKNOWN:
                unresolved.add(row[3])
    return len(unresolved)


def maybe_backup(path: Path, keep_backup: bool) -> None:
    if not keep_backup:
        return
    backup = path.with_suffix(path.suffix + ".bak")
    backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill [UNKNOWN_POI] names in enriched Yelp files (offline).")
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "Yelp" / "extendData"

    parser.add_argument("--home", type=Path, default=out_dir / "enriched_home.txt")
    parser.add_argument("--oot", type=Path, default=out_dir / "enriched_oot.txt")
    parser.add_argument("--cache", type=Path, default=out_dir / "yelp_business_cache.json")
    parser.add_argument("--keep-backup", action="store_true", help="Write .bak backups before overwriting.")
    args = parser.parse_args()

    home_rows = read_tsv(args.home)
    oot_rows = read_tsv(args.oot)

    name_map = load_cache_name_map(args.cache)
    name_map.update(extract_name_map_from_rows(home_rows))
    name_map.update(extract_name_map_from_rows(oot_rows))

    h_total_before, h_unknown_before = count_unknown(home_rows)
    o_total_before, o_unknown_before = count_unknown(oot_rows)

    home_new, h_filled = fill_unknown(home_rows, name_map)
    oot_new, o_filled = fill_unknown(oot_rows, name_map)

    h_total_after, h_unknown_after = count_unknown(home_new)
    o_total_after, o_unknown_after = count_unknown(oot_new)

    maybe_backup(args.home, args.keep_backup)
    maybe_backup(args.oot, args.keep_backup)
    write_tsv(args.home, home_new)
    write_tsv(args.oot, oot_new)

    unresolved_bids = unresolved_unique_bids(home_new, oot_new)

    print("Done.")
    print(f"name_map_size={len(name_map)}")
    print(f"home: total={h_total_before}, unknown_before={h_unknown_before}, filled={h_filled}, unknown_after={h_unknown_after}")
    print(f"oot : total={o_total_before}, unknown_before={o_unknown_before}, filled={o_filled}, unknown_after={o_unknown_after}")
    print(f"remaining_unknown_unique_bids={unresolved_bids}")


if __name__ == "__main__":
    main()
