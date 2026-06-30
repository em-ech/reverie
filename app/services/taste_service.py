"""Template taste blurb: one or two sentences summarizing a viewer's profile.

Pure template (no LLM), deterministic, so it always renders in the demo. Built
from the model's taste vector (top genres) plus the viewer's highest rated
films, phrased through the shared copy helpers (no hyphens).
"""

from __future__ import annotations

from app.services.copy import genre_label, join_list


def taste_blurb(taste: dict[str, float] | None, top_films: list[str]) -> str | None:
    """taste: {genre: weight} from rec.taste_vector. top_films: a few of the
    viewer's highest rated titles. Returns None when there is nothing to say."""
    if not taste:
        return None

    top = [genre_label(g) for g, _ in sorted(taste.items(), key=lambda x: x[1], reverse=True)[:3]]
    genres = join_list(top)
    lead = f"You lean toward {genres}"

    # top_films are the viewer's OWN highest-rated films (taste anchors), not
    # recommendations, so the phrasing frames them as favorites.
    if len(top_films) >= 2:
        return f"{lead}, going by favorites like {top_films[0]} and {top_films[1]}."
    if top_films:
        return f"{lead}, going by a favorite like {top_films[0]}."
    return f"{lead}. The picks below lean into exactly that."
