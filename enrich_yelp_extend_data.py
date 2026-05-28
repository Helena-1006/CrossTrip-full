#!/usr/bin/env python3
"""Enrich Yelp home/oot trajectories with POI name and category tree.

Input format (6 columns, tab-separated):
uid, cuid, city, business_id, timestamp, std_tag

Output format (8 columns, tab-separated):
uid, cuid, city, business_id, timestamp, std_tag, poi_name, poi_category_tree
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


# Fill this when you want to run directly without environment variables.
API_KEY_PLACEHOLDER = "2l5LwigHuNLzovjxC1I_IeSXpxmJnuO6Gt9fMh960T7VNnKm6kEqqXRnaAaUTsagcudqCtBh-bykIglJJ3FSiRzCaPBTDvo3iHIQHkzRHvxvfn_tGz3u5UQcX2u2aXYx"
API_BASE = "https://api.yelp.com/v3"


@dataclass
class BusinessInfo:
    name: str
    categories: List[Dict[str, str]]


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().replace("&", "and").split())


class YelpAPI:
    def __init__(self, api_key: str, sleep_seconds: float = 0.15, timeout: int = 20):
        self.api_key = api_key
        self.sleep_seconds = sleep_seconds
        self.timeout = timeout

    def _request_json(self, path: str, query: Optional[Dict[str, str]] = None) -> Dict:
        url = f"{API_BASE}{path}"
        if query:
            url = f"{url}?{urlencode(query)}"

        req = Request(url)
        req.add_header("Authorization", f"Bearer {self.api_key}")
        req.add_header("Accept", "application/json")

        try:
            with urlopen(req, timeout=self.timeout) as resp:
                payload = resp.read().decode("utf-8")
                time.sleep(self.sleep_seconds)
                return json.loads(payload)
        except HTTPError as e:
            msg = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"HTTP {e.code} for {url}: {msg[:300]}") from e
        except URLError as e:
            raise RuntimeError(f"Network error for {url}: {e}") from e

    def fetch_business(self, business_id: str) -> BusinessInfo:
        safe_id = quote(business_id)
        obj = self._request_json(f"/businesses/{safe_id}")
        name = obj.get("name") or "[UNKNOWN_POI]"
        categories = obj.get("categories") or []
        categories = [c for c in categories if isinstance(c, dict)]
        return BusinessInfo(name=name, categories=categories)

    def fetch_category_index(self, locale: str = "en_US") -> Dict[str, Dict]:
        obj = self._request_json("/categories", query={"locale": locale})
        index: Dict[str, Dict] = {}
        for c in obj.get("categories", []):
            alias = c.get("alias")
            if alias:
                index[alias] = c
        return index


def choose_best_business_category(categories: List[Dict[str, str]], std_tag: str) -> Optional[Dict[str, str]]:
    if not categories:
        return None

    tag_norm = normalize_text(std_tag)

    for c in categories:
        title = c.get("title", "")
        if normalize_text(title) == tag_norm:
            return c

    for c in categories:
        title = c.get("title", "")
        if tag_norm in normalize_text(title) or normalize_text(title) in tag_norm:
            return c

    return categories[0]


def build_category_tree(alias: Optional[str], fallback_title: str, category_index: Dict[str, Dict]) -> str:
    def _fallback_tree(title: str) -> str:
        t = normalize_text(title)

        if not title or t in ("", "unknown", "[unknown_poi]"):
            return "Other > Unknown"

        bar_tokens = (
            "bar", "bars", "pub", "speakeasy", "lounge", "nightclub", "beer", "wine"
        )
        if any(tok in t for tok in bar_tokens):
            return f"Dining and Drinking > Bar > {title}"

        cafe_tokens = ("coffee", "tea", "cafe", "cafes")
        if any(tok in t for tok in cafe_tokens):
            return f"Dining and Drinking > Cafe, Coffee, and Tea House > {title}"

        dessert_tokens = ("ice cream", "frozen yogurt", "dessert", "donut", "cupcake", "bakery")
        if any(tok in t for tok in dessert_tokens):
            return f"Dining and Drinking > Dessert Shop > {title}"

        if any(tok in t for tok in ("hotel", "resort", "lodging")):
            return "Travel and Transportation > Lodging > Hotel"

        if any(tok in t for tok in ("airport", "train", "station", "metro", "subway", "bus")):
            return f"Travel and Transportation > Transport Hub > {title}"

        food_tokens = (
            "restaurant", "restaurants", "burger", "pizza", "taco", "bbq", "sandwich",
            "seafood", "italian", "mexican", "indian", "japanese", "chinese", "thai",
            "vietnamese", "southern", "breakfast", "brunch", "diner", "food"
        )
        if any(tok in t for tok in food_tokens):
            return f"Dining and Drinking > Restaurant > {title}"

        if any(tok in t for tok in ("museum", "art", "theater", "cinema", "music", "festival", "zoo", "entertainment")):
            return f"Arts and Entertainment > {title}"

        if any(tok in t for tok in ("park", "beach", "plaza", "monument", "landmark", "garden", "trail", "hiking")):
            return f"Landmarks and Outdoors > {title}"

        if any(tok in t for tok in ("shop", "shopping", "store", "mall", "market", "bookstore", "grocery", "bakery")):
            return f"Retail > {title}"

        if any(tok in t for tok in ("salon", "spa", "massage", "medical", "dent", "doctor", "fitness", "gym", "yoga", "pilates")):
            return f"Health and Wellness > {title}"

        if any(tok in t for tok in ("office", "service", "services", "repair", "automotive", "car", "rental", "shipping", "event")):
            return f"Business and Professional Services > {title}"

        return f"Other > {title}"

    if not alias:
        return _fallback_tree(fallback_title)

    visited = set()

    def _path(a: str) -> List[str]:
        if not a or a in visited:
            return []
        visited.add(a)

        node = category_index.get(a)
        if not node:
            return []

        title = str(node.get("title") or "").strip()
        parents = node.get("parent_aliases") or []
        if not parents:
            return [title] if title else []

        parent_alias = parents[0]
        parent_path = _path(parent_alias)
        if title:
            return parent_path + [title]
        return parent_path

    path = [p for p in _path(alias) if p]
    if not path and fallback_title:
        return _fallback_tree(fallback_title)
    return " > ".join(path)


def read_rows(path: Path) -> List[List[str]]:
    rows: List[List[str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            rows.append(line.split("\t"))
    return rows


def write_rows(path: Path, rows: List[List[str]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write("\t".join(row) + "\n")


def ensure_api_key(cli_api_key: Optional[str]) -> str:
    candidates = [
        cli_api_key,
        os.getenv("YELP_API_KEY"),
        API_KEY_PLACEHOLDER,
    ]
    for key in candidates:
        if key and key != "YOUR_YELP_API_KEY_HERE":
            return key
    raise ValueError(
        "No Yelp API key found. Set --api-key, or export YELP_API_KEY, "
        "or edit API_KEY_PLACEHOLDER in this script."
    )


def load_cache(cache_path: Path) -> Dict[str, Dict]:
    if not cache_path.exists():
        return {}
    with cache_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_cache(cache_path: Path, cache: Dict[str, Dict]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def load_category_cache(path: Path) -> Dict[str, Dict]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_category_cache(path: Path, category_index: Dict[str, Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(category_index, f, ensure_ascii=False, indent=2)


def is_rate_limit_error(msg: str) -> bool:
    u = msg.upper()
    return "HTTP 429" in u or "ACCESS_LIMIT_REACHED" in u


def should_skip_cached_entry(entry: Dict, retry_failed: bool, refetch_all: bool) -> bool:
    if refetch_all:
        return False

    has_categories = bool(entry.get("categories"))
    has_valid_name = str(entry.get("name") or "") not in ("", "[UNKNOWN_POI]")
    has_error = bool(entry.get("error"))

    if has_categories or has_valid_name:
        return True
    if has_error and not retry_failed:
        return True
    return False


def classify_cache_entry(entry: Optional[Dict]) -> str:
    if not entry:
        return "missing"

    has_categories = bool(entry.get("categories"))
    has_valid_name = str(entry.get("name") or "") not in ("", "[UNKNOWN_POI]")
    if has_categories or has_valid_name:
        return "ok"

    err = str(entry.get("error") or "")
    if err:
        if is_rate_limit_error(err):
            return "failed_rate_limit"
        return "failed_other"
    return "empty"


def summarize_progress(
    business_ids: List[str],
    cache: Dict[str, Dict],
    retry_failed: bool,
    refetch_all: bool,
) -> Dict[str, int]:
    stats = {
        "total": len(business_ids),
        "ok": 0,
        "missing": 0,
        "empty": 0,
        "failed_rate_limit": 0,
        "failed_other": 0,
        "to_fetch_now": 0,
    }

    for bid in business_ids:
        entry = cache.get(bid)
        cls = classify_cache_entry(entry)
        stats[cls] += 1
        if not should_skip_cached_entry(entry or {}, retry_failed, refetch_all):
            stats["to_fetch_now"] += 1

    return stats


def print_progress_summary(label: str, stats: Dict[str, int]) -> None:
    total = max(stats["total"], 1)
    done_ratio = stats["ok"] / total * 100.0
    print(
        f"[summary:{label}] total={stats['total']} ok={stats['ok']} "
        f"missing={stats['missing']} empty={stats['empty']} "
        f"failed_rate_limit={stats['failed_rate_limit']} failed_other={stats['failed_other']} "
        f"to_fetch_now={stats['to_fetch_now']} done={done_ratio:.2f}%"
    )


def enrich_rows(
    rows: List[List[str]],
    business_map: Dict[str, BusinessInfo],
    category_index: Dict[str, Dict],
) -> List[List[str]]:
    out: List[List[str]] = []
    for row in rows:
        if len(row) < 6:
            continue

        uid, cuid, city, business_id, timestamp, std_tag = row[:6]
        info = business_map.get(business_id)

        if info is None:
            poi_name = "[UNKNOWN_POI]"
            poi_tree = build_category_tree(None, std_tag, category_index)
        else:
            poi_name = info.name
            best = choose_best_business_category(info.categories, std_tag)
            fallback = (best or {}).get("title", std_tag)
            alias = (best or {}).get("alias")
            poi_tree = build_category_tree(alias, fallback, category_index)
            if not poi_tree:
                poi_tree = std_tag

        out.append([uid, cuid, city, business_id, timestamp, std_tag, poi_name, poi_tree])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich Yelp home/oot with POI name + category tree")
    repo_root = Path(__file__).resolve().parents[1]
    yelp_dir = repo_root / "Yelp"
    default_out_dir = yelp_dir / "extendData"

    parser.add_argument("--api-key", type=str, default=None, help="Yelp API Key")
    parser.add_argument("--home-input", type=Path, default=yelp_dir / "home.txt")
    parser.add_argument("--oot-input", type=Path, default=yelp_dir / "oot.txt")
    parser.add_argument("--output-dir", type=Path, default=default_out_dir)
    parser.add_argument("--cache-path", type=Path, default=default_out_dir / "yelp_business_cache.json")
    parser.add_argument("--sleep-seconds", type=float, default=0.15)
    parser.add_argument("--locale", type=str, default="en_US")
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--refresh-categories", action="store_true",
                        help="Force refresh category index from Yelp API.")
    parser.add_argument("--retry-failed", action="store_true", default=True,
                        help="Retry cached failed entries (recommended for next-day resume).")
    parser.add_argument("--no-retry-failed", action="store_false", dest="retry_failed",
                        help="Do not retry cached failed entries.")
    parser.add_argument("--refetch-all", action="store_true",
                        help="Ignore cache and refetch all business IDs.")
    parser.add_argument("--stop-on-rate-limit", action="store_true", default=True,
                        help="Stop immediately when HTTP 429 / ACCESS_LIMIT_REACHED is hit.")
    parser.add_argument("--no-stop-on-rate-limit", action="store_false", dest="stop_on_rate_limit",
                        help="Continue even if rate limit is reached (not recommended).")
    parser.add_argument("--progress-only", action="store_true",
                        help="Only print progress summary and exit.")
    parser.add_argument("--offline-only", action="store_true",
                        help="Do not call Yelp API. Use local cache + offline rules only.")
    args = parser.parse_args()

    api_key = ensure_api_key(args.api_key)

    home_rows = read_rows(args.home_input)
    oot_rows = read_rows(args.oot_input)

    all_rows = home_rows + oot_rows
    business_ids = []
    seen = set()
    for row in all_rows:
        if len(row) < 4:
            continue
        bid = row[3]
        if bid not in seen:
            seen.add(bid)
            business_ids.append(bid)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = load_cache(args.cache_path)

    stats_before = summarize_progress(
        business_ids,
        cache,
        retry_failed=args.retry_failed,
        refetch_all=args.refetch_all,
    )
    print_progress_summary("before", stats_before)
    if args.progress_only:
        return

    api = YelpAPI(api_key=api_key, sleep_seconds=args.sleep_seconds)
    category_cache_path = args.output_dir / f"yelp_categories_{args.locale}.json"
    category_index: Dict[str, Dict] = {}

    if args.offline_only:
        if category_cache_path.exists():
            category_index = load_category_cache(category_cache_path)
            print(f"[info] offline mode: loaded category index cache: {category_cache_path}")
        else:
            category_index = {}
            print("[info] offline mode: no category index cache, using rule-based trees only.")
    elif (not args.refresh_categories) and category_cache_path.exists():
        category_index = load_category_cache(category_cache_path)
        print(f"[info] loaded category index from cache: {category_cache_path}")
    else:
        try:
            category_index = api.fetch_category_index(locale=args.locale)
            save_category_cache(category_cache_path, category_index)
            print(f"[info] fetched and cached category index: {category_cache_path}")
        except Exception as e:
            if category_cache_path.exists():
                category_index = load_category_cache(category_cache_path)
                print(f"[warn] category API unavailable, fallback to cached index: {e}")
            else:
                category_index = {}
                print(
                    "[warn] category API unavailable and no local category cache found. "
                    "Will continue with fallback category tree values."
                )

    fetched = 0
    failed = 0
    skipped_cached = 0
    stopped_by_rate_limit = False

    try:
        if args.offline_only:
            skipped_cached = len(business_ids)
            print("[info] offline mode: skipped all API fetch requests.")
        else:
            for idx, bid in enumerate(business_ids, start=1):
                entry = cache.get(bid)
                if entry and should_skip_cached_entry(entry, args.retry_failed, args.refetch_all):
                    skipped_cached += 1
                    if idx % args.save_every == 0:
                        print(
                            f"[progress] processed={idx}/{len(business_ids)} "
                            f"fetched={fetched} failed={failed} skipped_cached={skipped_cached}"
                        )
                    continue

                try:
                    info = api.fetch_business(bid)
                    cache[bid] = {
                        "name": info.name,
                        "categories": info.categories,
                    }
                    fetched += 1
                except Exception as e:
                    err = str(e)
                    if args.stop_on_rate_limit and is_rate_limit_error(err):
                        stopped_by_rate_limit = True
                        print("[warn] Yelp API rate limit reached (HTTP 429). Stop now and resume later.")
                        break

                    cache[bid] = {
                        "name": "[UNKNOWN_POI]",
                        "categories": [],
                        "error": err,
                    }
                    failed += 1

                if idx % args.save_every == 0:
                    save_cache(args.cache_path, cache)
                    print(
                        f"[progress] processed={idx}/{len(business_ids)} "
                        f"fetched={fetched} failed={failed} skipped_cached={skipped_cached}"
                    )
    except KeyboardInterrupt:
        print("[warn] Interrupted by user. Saving progress before exit...")
    finally:
        save_cache(args.cache_path, cache)

    business_map: Dict[str, BusinessInfo] = {}
    for bid, obj in cache.items():
        business_map[bid] = BusinessInfo(
            name=str(obj.get("name") or "[UNKNOWN_POI]"),
            categories=obj.get("categories") or [],
        )

    enriched_home = enrich_rows(home_rows, business_map, category_index)
    enriched_oot = enrich_rows(oot_rows, business_map, category_index)

    home_out = args.output_dir / "enriched_home.txt"
    oot_out = args.output_dir / "enriched_oot.txt"
    write_rows(home_out, enriched_home)
    write_rows(oot_out, enriched_oot)

    print("Done.")
    print(f"home rows: {len(enriched_home)} -> {home_out}")
    print(f"oot rows: {len(enriched_oot)} -> {oot_out}")
    print(
        f"fetched businesses: {fetched}, failed: {failed}, "
        f"skipped_cached: {skipped_cached}, cache: {args.cache_path}"
    )
    if stopped_by_rate_limit:
        print("Stopped due to rate limit. Re-run later with the same command to resume.")

    stats_after = summarize_progress(
        business_ids,
        cache,
        retry_failed=args.retry_failed,
        refetch_all=args.refetch_all,
    )
    print_progress_summary("after", stats_after)


if __name__ == "__main__":
    main()
