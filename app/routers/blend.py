"""The Blend endpoint: a Spotify-Blend-style merge of two friends' tastes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import get_db
from app.deps import get_current_user
from app.models import User
from app.services import blend_service

router = APIRouter(tags=["blend"])


@router.get("/blend/{friend_id}")
def get_blend(
    friend_id: int,
    n: int = 12,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    try:
        return blend_service.blend(db, user.id, friend_id, n=n)
    except blend_service.BlendError as e:
        raise HTTPException(status_code=e.status, detail=e.detail)
