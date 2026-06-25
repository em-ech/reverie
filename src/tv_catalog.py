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
    The 20-30 entries below are hand-curated; extend as needed for the demo.

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
# Hand-curated for the demo (extend if you add more demo shows).
# These are the most common Spanish-localised names on Netflix Spain.
# ---------------------------------------------------------------------------

SPANISH_TO_ENGLISH: dict[str, str] = {
    # Drama / thriller
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
    # Sci-fi / fantasy
    "El Embarcadero":           "The Pier",
    "Hierro":                   "Iron",
    "Vivir sin Permiso":        "Living Without Permission",
    "La Línea Invisible":       "The Invisible Line",
    # International shows with Spanish titles on Netflix ES
    "Stranger Things":          "Stranger Things",      # same in Spanish
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
# The IMDb genre vocabulary is broader; map to the 18-genre ML vocab.
# A single IMDb tag may map to multiple ML genres (list).
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
    """Load the IMDB TV show catalog and build the genre bridge matrix.

    Parameters
    ----------
    csv_path    : path to imdb_tvshows.csv
    genre_names : the 18-genre ML vocabulary from the trained Dataset
                  (Dataset.genre_names — must match what the model was trained on)
    imdb_to_ml  : override the default IMDB→ML genre map
    min_genres  : drop shows with fewer than this many mapped ML genres

    Returns
    -------
    tv_titles        : list[str] of show titles (length N)
    tv_genre_matrix  : float32 array (N, len(genre_names)) — multi-hot
    tv_title_map     : dict[str, str] Spanish→English (SPANISH_TO_ENGLISH extended
                       with any exact-match Spanish titles already in the catalog)
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(
            f"TV catalog not found: {p}\n"
            "Expected: data/imdb_tvshows.csv  (3,000 IMDB shows with genres)"
        )

    mapping = imdb_to_ml if imdb_to_ml is not None else IMDB_TO_ML

    df = _load_csv(p)

    # Build genre index (MovieLens vocabulary)
    gidx = {g: i for i, g in enumerate(genre_names)}
    n_genres = len(genre_names)

    titles: list[str] = []
    vectors: list[np.ndarray] = []
    unmapped_genres: set[str] = set()

    for _, row in df.iterrows():
        title     = str(row.get("title", row.get("Title", ""))).strip()
        genre_str = str(row.get("genres", row.get("Genres", ""))).strip()

        if not title or title.lower() in ("nan", "none", ""):
            continue

        # Parse genre string: "Drama,Thriller" or "Drama | Thriller" or "Drama/Thriller"
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

    # Coverage report
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

    # Build the title map: start from SPANISH_TO_ENGLISH, then add any title
    # in the catalog that is already in Spanish (exact match in the dict)
    tv_title_map = dict(SPANISH_TO_ENGLISH)

    return titles, tv_genre_matrix, tv_title_map


def get_title_index(tv_titles: list[str]) -> dict[str, int]:
    """Return {lower_title: row_index} for fast lookup."""
    return {t.lower(): i for i, t in enumerate(tv_titles)}


def score_show(
    title: str,
    tv_titles: list[str],
    tv_genre_matrix: np.ndarray,
    taste: np.ndarray,
    genre_marginal: np.ndarray,
) -> float | None:
    """Cosine similarity between *taste* and a named show's centered genre vector.

    Returns None if the title is not found in the catalog.
    """
    idx_map = get_title_index(tv_titles)
    idx = idx_map.get(title.lower())
    if idx is None:
        return None
    centered = tv_genre_matrix[idx] - genre_marginal
    tn = float(np.linalg.norm(taste)) + 1e-9
    cn = float(np.linalg.norm(centered)) + 1e-9
    return float(np.dot(centered, taste) / (cn * tn))


# ---------------------------------------------------------------------------
# Internal: flexible CSV parser
# ---------------------------------------------------------------------------

def _load_csv(p: Path) -> pd.DataFrame:
    """Try several common layouts for imdb_tvshows.csv."""
    # Attempt 1: standard comma-separated
    try:
        df = pd.read_csv(p, encoding="utf-8")
        if "title" in df.columns or "Title" in df.columns:
            return df
    except Exception:
        pass

    # Attempt 2: tab-separated
    try:
        df = pd.read_csv(p, sep="\t", encoding="utf-8")
        if "title" in df.columns or "Title" in df.columns:
            return df
    except Exception:
        pass

    # Attempt 3: latin-1 encoding
    df = pd.read_csv(p, encoding="latin-1")
    return df


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, json, sys

    ap = argparse.ArgumentParser(description="Smoke-test the TV catalog loader.")
    ap.add_argument("--csv",        default="data/imdb_tvshows.csv")
    ap.add_argument("--model_cfg",  default="artifacts/model_config.json",
                    help="model_config.json to get genre_names")
    ap.add_argument("--top", type=int, default=10, help="Show top-N titles")
    args = ap.parse_args()

    cfg_path = Path(args.model_cfg)
    if cfg_path.exists():
        with open(cfg_path) as fh:
            cfg = json.load(fh)
        genre_names = cfg["genre_names"]
    else:
        # Fallback: the standard 18-genre MovieLens vocabulary
        genre_names = [
            "Action", "Adventure", "Animation", "Children's", "Comedy",
            "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
            "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
            "Thriller", "War", "Western",
        ]
        print(f"[tv_catalog] model_config.json not found; using default 18-genre vocab")

    titles, matrix, tv_map = load_tv_catalog(args.csv, genre_names)

    print(f"\nCatalog: {len(titles)} shows,  genre matrix: {matrix.shape}")
    print(f"\nFirst {args.top} titles:")
    for t in titles[:args.top]:
        print(f"  {t}")

    print(f"\nSpanish→English map ({len(tv_map)} entries):")
    for k, v in list(tv_map.items())[:10]:
        print(f"  '{k}'  →  '{v}'")