"""The single seam between the persisted account layer and the frozen model.

Turns a user's stored history into the ordered (movieId, stars) sequence the
GRU consumes. Imported by the history and blend routers/services so nothing
else touches src/recommend directly.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import HistoryItem


def ordered_history(db: Session, user_id: int) -> list[tuple[int, float]]:
    """The user's history as the position-ordered (movieId, stars) sequence."""
    rows = db.scalars(
        select(HistoryItem)
        .where(HistoryItem.user_id == user_id)
        .order_by(HistoryItem.position)
    ).all()
    return [(r.movie_id, r.rating) for r in rows]
