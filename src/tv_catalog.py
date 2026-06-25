"""TV show catalog for the Reverie recommendation bridge.

Loads `data/imdb_tvshows.csv` (3,000 shows with genres + descriptions) and
maps each show's genre tags to the 18-genre MovieLens vocabulary so the
content-cosine bridge in `src/recommend.score_tv()` can score them.

Also provides the Spanish-title → English-title map used by the Netflix importer
for ~20-30 curated demo shows.

Key exports
-----------
load_tv_catalog(csv_path, genre_names)
    Returns (tv_titles, tv_genre_matrix, tv_title_map).

SPANISH_TO_ENGLISH
    Dict of known Spanish Netflix localizations → canonical English titles.

build_tv_genre_matrix(df, genre_names, imdb_to_ml)
    Build the (n_shows, n_genres) float32 matrix from the parsed catalog.

Audit tags: [M4] shared genre vocabulary, ≥85% coverage target.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Spanish Netflix title → English canonical title
# ---------------------------------------------------------------------------

SPANISH_TO_ENGLISH: dict[str, str] = {
    "La Casa de Papel":         "Money Heist",
    "Élite":                    "Elite",
    "Vis a Vis":                "Locked Up",
    "El Ministerio del Tiempo": "The Ministry of Time",
    "Merli":                    "Merlí",
    "Paquita Salas":            "Paquita Salas",
    "Velvet":                   "Velvet",
    "Gran Hotel":               "Grand Hotel",
    "El Internado":             "The Boarding School",
    "Fariña":                   "Cocaine Coast",
    "El Embarcadero":           "The Pier",
    "Hierro":                   "Iron",
    "Vivir sin Permiso":        "Living Without Permission",
    "La Línea Invisible":       "The Invisible Line",
    "Stranger Things":          "Stranger Things",
    "The Crown":                "The Crown",
    "Narcos":                   "Narcos",
    "Merlí: Sapere Aude":       "Merlí: Sapere Aude",
    "Las Chicas del Cable":     "Cable Girls",
    "El Inocente":              "The Innocent",
    "Osmosis":                  "Osmosis",
    "Valeria":                  "Valeria",
    "Hache":                    "Hache",
    "Reyes de la Noche":        "Night Raiders",
    "La Peste":                 "The Plague",
    "Desaparecidos":            "Vanished",
    "Bajo Sospecha":            "Under Suspicion",
    "Tiempos de Guerra":        "Tiempos de Guerra",
    "Skam España":              "Skam España",
}


# ---------------------------------------------------------------------------
# IMDb genre string → MovieLens genre(s)
# ---------------------------------------------------------------------------

IMDB_TO_ML: dict[str, list[str]] = {
    "Action":          ["Action"],
    "Adventure":       ["Adventure"],
    "Animation":       ["Animation", "Children's"],
    "Biography":       ["Documentary"],
    "Children":        ["Children's"],
    "Comedy":          ["Comedy"],
    "Crime":           ["Crime"],
    "Documentary":     ["Documentary"],
    "Drama":           ["Drama"],
    "Family":          ["Children's"],
    "Fantasy":         ["Fantasy"],
    "Film-Noir":       ["Film-Noir"],
    "Game-Show":       ["Comedy"],
    "History":         ["Documentary"],
    "Horror":          ["Horror"],
    "Music":           ["Musical"],
    "Musical":         ["Musical"],
    "Mystery":         ["Mystery"],
    "News":            ["Documentary"],
    "Reality-TV":      ["Documentary"],
    "Romance":         ["Romance"],
    "Sci-Fi":          ["Sci-Fi"],
    "Short":           ["Drama"],
    "Sport":           ["Documentary"],
    "Talk-Show":       ["Comedy"],
    "Thriller":        ["Thriller"],
    "War":             ["War"],
    "Western":         ["Western"],
}


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

def load_tv_catalog(
    csv_path: str,
    genre_names: list[str],
    imdb_to_ml: dict[str, list[str]] | None = None,
    min_genres: int = 1,
) -> tuple[list[str], np.ndarray, dict[str, str]]:
    """Load the IMDB TV show catalog and build the genre bridge matrix."""
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(
            f"TV catalog not found: {p}\n"
            "Expected: data/imdb_tvshows.csv  (3,000 IMDB shows with genres)"
        )

    mapping = imdb_to_ml if imdb_to_ml is not None else IMDB_TO_ML
    df = _load_csv(p)

    gidx = {g: i for i, g in enumerate(genre_names)}
    n_genres = len(genre_names)

    titles: list[str] = []
    vectors: list[np.ndarray] = []
    unmapped_genres: set[str] = set()

    for _, row in df.iterrows():
        title     = str(row.get("Title", row.get("title", ""))).strip()
        genre_str = str(row.get("Genres", row.get("genres", ""))).strip()

        if not title or title.lower() in ("nan", "none", ""):
            continue

        raw_genres = re.split(r"[,|/;]+", genre_str)
        raw_genres = [g.strip() for g in raw_genres if g.strip()]

        vec = np.zeros(n_genres, dtype=np.float32)
        matched = False
        for rg in raw_genres:
            ml_genres = mapping.get(rg, [])
            if not ml_genres:
                unmapped_genres.add(rg)
            for mlg in ml_genres:
                idx = gidx.get(mlg)
                if idx is not None:
                    vec[idx] = 1.0
                    matched = True

        if matched and int(vec.sum()) >= min_genres:
            titles.append(title)
            vectors.append(vec)

    tv_genre_matrix = np.stack(vectors, axis=0) if vectors else np.zeros((0, n_genres), dtype=np.float32)

    total_in_file = len(df)
    coverage = 100 * len(titles) / max(total_in_file, 1)
    print(
        f"[tv_catalog] {total_in_file} shows in file → "
        f"{len(titles)} with ≥{min_genres} mapped genre(s) "
        f"({coverage:.0f}% coverage)"
    )
    if coverage < 85:
        print(
            f"[tv_catalog] WARNING: coverage {coverage:.0f}% is below the 85% target  [M4]\n"
            f"  Unmapped IMDb genre tags: {sorted(unmapped_genres)}\n"
            f"  Add them to IMDB_TO_ML in src/tv_catalog.py"
        )
    else:
        print(f"[tv_catalog] Coverage ≥85% target met  [M4] ✓")

    if unmapped_genres:
        print(f"[tv_catalog] Unmapped genre tags (not blocking): {sorted(unmapped_genres)}")

    tv_title_map = dict(SPANISH_TO_ENGLISH)
    return titles, tv_genre_matrix, tv_title_map


def get_title_index(tv_titles: list[str]) -> dict[str, int]:
    return {t.lower(): i for i, t in enumerate(tv_titles)}


def score_show(
    title: str,
    tv_titles: list[str],
    tv_genre_matrix: np.ndarray,
    taste: np.ndarray,
    genre_marginal: np.ndarray,
) -> float | None:
    idx_map = get_title_index(tv_titles)
    idx = idx_map.get(title.lower())
    if idx is None:
        return None
    centered = tv_genre_matrix[idx] - genre_marginal
    tn = float(np.linalg.norm(taste)) + 1e-9
    cn = float(np.linalg.norm(centered)) + 1e-9
    return float(np.dot(centered, taste) / (cn * tn))


# ---------------------------------------------------------------------------
# Internal: CSV parser for the specific imdb_tvshows.csv format
# The file wraps each row in outer quotes and uses "" for internal quoting,
# with trailing ;;; on every line.
# ---------------------------------------------------------------------------

def _load_csv(p: Path) -> pd.DataFrame:
    with open(p, encoding="utf-8", errors="replace") as fh:
        raw = fh.read()

    # Strip trailing semicolons from line endings
    cleaned = re.sub(r";+\r?\n", "\n", raw).strip()
    cleaned = re.sub(r";+$", "", cleaned)

    records = []
    for line in cleaned.split("\n")[1:]:   # skip header row
        line = line.strip()
        if not line:
            continue
        # Remove outer wrapping quotes
        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1]

        # Extract all ""-quoted fields (About, Genres, Actors use this escaping)
        quoted_fields = re.findall(r'""([^""]+)""', line)

        # Extract title: everything before the first "" block
        title_match = re.match(r'^(.+?),""', line)
        if title_match:
            title = title_match.group(1).strip()
            # If unquoted About text leaked in (contains comma), keep only first segment
            if "," in title:
                title = title.split(",")[0].strip()
        else:
            title = line.split(",")[0].strip()

        # Identify genres among the quoted fields:
        # Genres = short field, no sentence punctuation, all tokens ≤ 2 words
        genres = None
        for qf in quoted_fields:
            if "." not in qf and len(qf) < 80 and not any(c.isdigit() for c in qf):
                words = [w.strip() for w in qf.split(",")]
                if words and all(len(w.split()) <= 2 for w in words):
                    genres = qf
                    break

        if title and genres:
            records.append({"Title": title, "Genres": genres})

    if len(records) > 100:
        return pd.DataFrame(records)

    # Fallback: standard pandas
    for enc in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(p, encoding=enc, engine="python", on_bad_lines="skip")
            if "Title" in df.columns or "title" in df.columns:
                return df
        except Exception:
            continue

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, json, sys

    ap = argparse.ArgumentParser(description="Smoke-test the TV catalog loader.")
    ap.add_argument("--csv",       default="data/imdb_tvshows.csv")
    ap.add_argument("--model_cfg", default="artifacts/model_config.json")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    cfg_path = Path(args.model_cfg)
    if cfg_path.exists():
        with open(cfg_path) as fh:
            cfg = json.load(fh)
        genre_names = cfg["genre_names"]
    else:
        genre_names = [
            "Action", "Adventure", "Animation", "Children's", "Comedy",
            "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
            "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
            "Thriller", "War", "Western",
        ]
        print("[tv_catalog] model_config.json not found; using default 18-genre vocab")

    titles, matrix, tv_map = load_tv_catalog(args.csv, genre_names)

    print(f"\nCatalog: {len(titles)} shows,  genre matrix: {matrix.shape}")
    print(f"\nFirst {args.top} titles:")
    for t in titles[:args.top]:
        print(f"  {t}")

    print(f"\nSpanish→English map ({len(tv_map)} entries):")
    for k, v in list(tv_map.items())[:10]:
        print(f"  '{k}'  →  '{v}'")
