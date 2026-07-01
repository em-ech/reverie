"""The Blend: run the RNN for two friends and surface what they'd both love.

watch_together = the movies on BOTH recommendation lists, ranked by the
harmonic mean of the two scores (rewards films both models rank highly).
Also returns the histories' intersection ("already in common") and the
averaged taste vector with a blurb.
"""

from __future__ import annotations

import numpy as np
from sqlalchemy.orm import Session

from app import enrich
from app.models import User
from app.services import friend_service as fs
from app.services.blurb_service import blend_blurb
from app.services.recommend_service import ordered_history
from src.ncf_model import recommend as ncf
from src.gru_model import recommend as rec

BLEND_POOL = 60  # wide per-person pull so the intersection has material


def _modern_taste_vec(history, genre_names) -> np.ndarray:
    """Rating-weighted genre proportions aligned to genre_names (modern mode)."""
    prof = ncf.taste_genres(history, lambda mid: enrich.enrich(mid)["genres"])
    return np.array([prof.get(g, 0.0) for g in genre_names], dtype=float)


class BlendError(Exception):
    def __init__(self, status: int, detail: str):
        self.status = status
        self.detail = detail
        super().__init__(detail)


def _harmonic(a: float, b: float) -> float:
    return 2 * a * b / (a + b) if (a + b) > 0 else 0.0


def _vec_dict(vec, genre_names) -> dict[str, float]:
    return {g: round(float(x), 4) for g, x in zip(genre_names, vec)}


def _brief(user: User) -> dict:
    return {"id": user.id, "username": user.username, "display_name": user.display_name}


def blend(db: Session, me_id: int, friend_id: int, n: int = 12) -> dict:
    if not fs.are_friends(db, me_id, friend_id):
        raise BlendError(403, "You can only blend with a friend")

    me = db.get(User, me_id)
    friend = db.get(User, friend_id)
    if me is None or friend is None:
        raise BlendError(404, "User not found")

    me_hist = ordered_history(db, me_id)
    fr_hist = ordered_history(db, friend_id)

    # Cold-start popularity recs are meaningless to blend — require both to have
    # built a history.
    if not me_hist or not fr_hist:
        missing = friend.username if not fr_hist else "You"
        detail = (
            f"{friend.username} hasn't built a watch history yet."
            if not fr_hist
            else "Build your watch history first."
        )
        return {
            "me": _brief(me),
            "friend": _brief(friend),
            "watch_together": [],
            "already_in_common": [],
            "taste": None,
            "blurb": detail,
            "degraded": "no_history",
        }

    if enrich.is_modern():
        me_recs = dict(ncf.rank_for_history(me_hist, n=BLEND_POOL))
        fr_recs = dict(ncf.rank_for_history(fr_hist, n=BLEND_POOL))
        genre_names = ncf.genre_names()
        t_me = _modern_taste_vec(me_hist, genre_names)
        t_fr = _modern_taste_vec(fr_hist, genre_names)
        match_fn = enrich.match_from_ratings
    else:
        me_recs = dict(rec.recommend_movies(me_hist, n=BLEND_POOL))
        fr_recs = dict(rec.recommend_movies(fr_hist, n=BLEND_POOL))
        genre_names = rec.load()["cfg"]["genre_names"]
        t_me = np.asarray(rec.taste_vector(me_hist))
        t_fr = np.asarray(rec.taste_vector(fr_hist))
        match_fn = enrich.match_scores

    inter = set(me_recs) & set(fr_recs)
    ranked = sorted(
        ((mid, _harmonic(me_recs[mid], fr_recs[mid])) for mid in inter),
        key=lambda x: x[1],
        reverse=True,
    )

    degraded: str | bool = False
    if not ranked:
        # No overlap in the top pools: fall back to the union by best single
        # score so there's still a shared list, and flag it.
        degraded = "no_overlap"
        union = {**fr_recs, **me_recs}
        ranked = sorted(union.items(), key=lambda x: x[1], reverse=True)

    top = ranked[:n]
    matches = match_fn([s for _, s in top])
    watch_together = [
        {**enrich.enrich(mid, s), "match": m} for (mid, s), m in zip(top, matches)
    ]

    # Movies both have already watched.
    common_ids = {mid for mid, _ in me_hist} & {mid for mid, _ in fr_hist}
    already_in_common = [enrich.enrich(mid) for mid in common_ids]

    # Blended taste = average of the two genre vectors.
    t_blend = (np.asarray(t_me) + np.asarray(t_fr)) / 2
    taste = {
        "me": _vec_dict(t_me, genre_names),
        "friend": _vec_dict(t_fr, genre_names),
        "blend": _vec_dict(t_blend, genre_names),
    }

    blurb = blend_blurb(friend.username, t_blend, genre_names, len(inter))

    return {
        "me": _brief(me),
        "friend": _brief(friend),
        "watch_together": watch_together,
        "already_in_common": already_in_common,
        "taste": taste,
        "blurb": blurb,
        "degraded": degraded,
    }
