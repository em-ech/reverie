"""Seed a Reverie demo profile (account + watch history) against the local API.

Build profiles for real people so the Blend has material. The model's catalog is
MovieLens ml-1m (movies up to the year 2000), so only pre-2001 titles resolve.

The API must be running (uvicorn app.api:app --port 8000).

Usage
-----
    # By favorite titles (comma separated; pre-2001 films)
    python -m scripts.seed_profile --username sara --display Sara \
        --titles "The Matrix, GoodFellas, Alien, Blade Runner, The Shining"

    # By a Letterboxd or Netflix export
    python -m scripts.seed_profile --username alex --letterboxd ~/exports/ratings.csv
    python -m scripts.seed_profile --username alex --netflix ~/exports/ViewingActivity.csv

Re-running for an existing username logs in and appends (history is idempotent
per movie). Every seeded profile shares the same demo password unless you pass
--password.
"""

from __future__ import annotations

import argparse
import sys

import requests

from src.catalog.importers import _normalise_title

API = "http://localhost:8000"
DEFAULT_PASSWORD = "reverie-demo"


def _register_or_login(username: str, password: str, display: str) -> str:
    r = requests.post(
        f"{API}/auth/register",
        json={"username": username, "password": password, "display_name": display},
    )
    if r.status_code == 200:
        print(f"[seed] created account '{username}'")
        return r.json()["token"]
    r = requests.post(f"{API}/auth/login", json={"username": username, "password": password})
    if r.status_code == 200:
        print(f"[seed] using existing account '{username}'")
        return r.json()["token"]
    sys.exit(f"[seed] could not register or log in '{username}': {r.text}")


def _resolve_title(title: str) -> dict | None:
    """Best-effort title to catalog movie, with article-aware matching
    ('The Matrix' resolves to MovieLens 'Matrix, The')."""
    norm = _normalise_title(title)  # article reordered, e.g. "matrix the"
    key = norm.split()[0] if norm else title
    results = requests.get(f"{API}/catalog", params={"q": key, "limit": 25}).json()
    for m in results:
        if _normalise_title(m["title"]) == norm:
            return m
    # Fall back to a raw substring search.
    results = requests.get(f"{API}/catalog", params={"q": title, "limit": 25}).json()
    return results[0] if results else None


def _add(token: str, movie_id: int, rating: float) -> None:
    requests.post(
        f"{API}/me/history",
        headers={"Authorization": f"Bearer {token}"},
        json={"movieId": movie_id, "rating": rating},
    )


def _seed_titles(token: str, titles: list[str], rating: float) -> tuple[list[str], list[str]]:
    matched, missed = [], []
    for raw in titles:
        title = raw.strip()
        if not title:
            continue
        m = _resolve_title(title)
        if m:
            _add(token, m["movieId"], rating)
            matched.append(m["title"])
        else:
            missed.append(title)
    return matched, missed


def _seed_csv(token: str, path: str, source: str) -> tuple[list[str], int]:
    with open(path, "rb") as fh:
        r = requests.post(f"{API}/import", files={"file": fh}, data={"source": source})
    if r.status_code != 200:
        sys.exit(f"[seed] import failed: {r.text}")
    data = r.json()
    for item in data["history"]:
        _add(token, item["movieId"], item["rating"])
    return [h["title"] for h in data["history"]], data["total"] - data["matched"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Seed a Reverie demo profile.")
    ap.add_argument("--username", required=True)
    ap.add_argument("--display", help="Display name (defaults to username)")
    ap.add_argument("--password", default=DEFAULT_PASSWORD)
    ap.add_argument("--titles", help="Comma-separated favorite titles (pre-2001)")
    ap.add_argument("--rating", type=float, default=4.5, help="Stars for --titles entries")
    ap.add_argument("--letterboxd", help="Path to a Letterboxd ratings.csv")
    ap.add_argument("--netflix", help="Path to a Netflix ViewingActivity.csv")
    args = ap.parse_args()

    token = _register_or_login(args.username, args.password, args.display or args.username)

    if args.titles:
        matched, missed = _seed_titles(token, args.titles.split(","), args.rating)
    elif args.letterboxd:
        matched, missed = _seed_csv(token, args.letterboxd, "letterboxd")
    elif args.netflix:
        matched, missed = _seed_csv(token, args.netflix, "netflix")
    else:
        sys.exit("[seed] provide one of --titles, --letterboxd, or --netflix")

    print(f"[seed] '{args.username}': added {len(matched)} movies")
    for t in matched:
        print(f"   + {t}")
    if isinstance(missed, list):
        if missed:
            print(f"[seed] {len(missed)} not in the catalog (pre-2001 only): {', '.join(missed)}")
    elif missed:
        print(f"[seed] {missed} rows not matched (post-2000 or not in ml-1m)")
    print(f"[seed] done. Login: {args.username} / {args.password}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
