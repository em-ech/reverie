"""Offline TMDB watch-providers fetcher for the Reverie demo.

For every movie that has a cached TMDB id (artifacts/posters.json), fetch where
it streams via TMDB's /watch/providers endpoint (powered by JustWatch) and cache
the US + ES "flatrate" (subscription streaming) providers to
artifacts/providers.json, so the API/UI can show "Stream on Netflix / Prime ..."
with no API key at serve time.

One-time build step (mirrors build_posters.py); never touches the model.

Usage
-----
    conda activate deep-learning
    TMDB_API_KEY=xxxxxxxx python -m src.catalog.build_providers
    # resumable: re-run to fill in any movies still missing

Output (artifacts/providers.json), keyed by MovieLens movieId:
    { "1": {"US": {"link": "https://...", "flatrate": [{"name": "Disney Plus",
                                                        "logo": "https://image.tmdb.org/t/p/w92/...jpg"}]},
            "ES": { ... }},
      ... }
A movie absent from a region simply omits that region key.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests

POSTERS_JSON = "artifacts/posters.json"
OUT_PATH = "artifacts/providers.json"
PROVIDERS_URL = "https://api.themoviedb.org/3/movie/{tmdb_id}/watch/providers"
LOGO_BASE = "https://image.tmdb.org/t/p/w92"
REGIONS = ("US", "ES")

PACING_SECONDS = 0.28
MAX_RETRIES = 3
CHECKPOINT_EVERY = 200


def _request(session: requests.Session, url: str, api_key: str) -> dict | None:
    """GET a TMDB endpoint with rate-limit + backoff handling."""
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(url, params={"api_key": api_key}, timeout=15)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 429:
            time.sleep(int(r.headers.get("Retry-After", "2")) + 1)
            continue
        if r.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        if r.status_code != 200:
            return None
        return r.json()
    return None


def _region_slice(region: dict) -> dict | None:
    """Normalize one TMDB region block to {link, flatrate:[{name, logo}]}."""
    flatrate = [
        {"name": p.get("provider_name"), "logo": f"{LOGO_BASE}{p.get('logo_path')}"}
        for p in region.get("flatrate", [])
        if p.get("logo_path")
    ]
    if not flatrate:
        return None
    return {"link": region.get("link"), "flatrate": flatrate}


def _save(cache: dict, path: Path) -> None:
    path.write_text(json.dumps(cache, indent=0, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    api_key = os.environ.get("TMDB_API_KEY")
    if not api_key:
        print("ERROR: set TMDB_API_KEY (themoviedb.org -> Settings -> API).", file=sys.stderr)
        return 1

    posters = json.loads(Path(POSTERS_JSON).read_text(encoding="utf-8"))
    # movieId -> tmdb_id, only for movies we have a TMDB id for.
    targets = {mid: v["tmdb_id"] for mid, v in posters.items() if v.get("tmdb_id")}

    out_path = Path(OUT_PATH)
    cache: dict[str, dict] = {}
    if out_path.exists():
        cache = json.loads(out_path.read_text(encoding="utf-8"))

    todo = [(mid, tid) for mid, tid in targets.items() if mid not in cache]
    print(f"{len(targets)} movies with TMDB ids; {len(cache)} cached; {len(todo)} to fetch.")

    session = requests.Session()
    hits = sum(1 for v in cache.values() if v)
    for i, (mid, tmdb_id) in enumerate(todo, 1):
        data = _request(session, PROVIDERS_URL.format(tmdb_id=tmdb_id), api_key)
        results = (data or {}).get("results", {})
        entry: dict = {}
        for region in REGIONS:
            sliced = _region_slice(results.get(region, {}))
            if sliced:
                entry[region] = sliced
        cache[mid] = entry
        if entry:
            hits += 1
        if i % CHECKPOINT_EVERY == 0:
            _save(cache, out_path)
            print(f"  {i}/{len(todo)} fetched, {hits} streamable (checkpoint saved)")
        time.sleep(PACING_SECONDS)

    _save(cache, out_path)
    streamable = sum(1 for v in cache.values() if v)
    print(f"Done. {len(cache)} movies cached, {streamable} streamable somewhere "
          f"(US/ES). -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
