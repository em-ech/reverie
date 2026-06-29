"""Per-user saved watch history (the 'Movies I've seen' section) + a
convenience recommend endpoint that runs on the saved history."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app import enrich
from app.db import get_db
from app.deps import get_current_user
from app.models import HistoryItem, User
from app.schemas import HistoryAddRequest, HistoryRatingRequest
from app.services.recommend_service import ordered_history
from src import recommend as rec

router = APIRouter(prefix="/me", tags=["history"])


def _enriched_history(db: Session, user_id: int) -> list[dict]:
    rows = db.scalars(
        select(HistoryItem).where(HistoryItem.user_id == user_id).order_by(HistoryItem.position)
    ).all()
    return [{**enrich.enrich(r.movie_id), "rating": r.rating, "position": r.position} for r in rows]


@router.get("/history")
def get_history(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    return {"history": _enriched_history(db, user.id)}


@router.post("/history")
def add_history(
    req: HistoryAddRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    if req.movieId not in enrich.recommendable_ids():
        raise HTTPException(status_code=400, detail="Movie not in the recommendable catalog")
    existing = db.scalar(
        select(HistoryItem).where(
            HistoryItem.user_id == user.id, HistoryItem.movie_id == req.movieId
        )
    )
    if existing:  # idempotent: re-adding just updates the rating
        existing.rating = req.rating
    else:
        next_pos = db.scalar(
            select(func.coalesce(func.max(HistoryItem.position), -1)).where(
                HistoryItem.user_id == user.id
            )
        )
        db.add(HistoryItem(
            user_id=user.id, movie_id=req.movieId, rating=req.rating, position=next_pos + 1,
        ))
    db.commit()
    return {"history": _enriched_history(db, user.id)}


@router.put("/history/{movie_id}")
def update_rating(
    movie_id: int,
    req: HistoryRatingRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    item = db.scalar(
        select(HistoryItem).where(
            HistoryItem.user_id == user.id, HistoryItem.movie_id == movie_id
        )
    )
    if item is None:
        raise HTTPException(status_code=404, detail="Not in your history")
    item.rating = req.rating
    db.commit()
    return {"history": _enriched_history(db, user.id)}


@router.delete("/history/{movie_id}")
def remove_history(
    movie_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    db.execute(
        delete(HistoryItem).where(
            HistoryItem.user_id == user.id, HistoryItem.movie_id == movie_id
        )
    )
    db.commit()
    return {"history": _enriched_history(db, user.id)}


@router.get("/recommend")
def recommend_from_history(
    n: int = 12,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    history = ordered_history(db, user.id)
    recs = rec.recommend_movies(history, n=n)
    matches = enrich.match_scores([s for _, s in recs])
    taste = rec.taste_vector(history) if history else None
    st = rec.load()
    return {
        "recommendations": [
            {**enrich.enrich(mid, s), "match": m} for (mid, s), m in zip(recs, matches)
        ],
        "taste": None if taste is None else {
            g: round(float(v), 4) for g, v in zip(st["cfg"]["genre_names"], taste)
        },
    }
