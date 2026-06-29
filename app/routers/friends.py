"""Friends: user search, friend requests, and the friends list."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import get_db
from app.deps import get_current_user
from app.models import User
from app.schemas import FriendRequestBody
from app.services import friend_service as fs

router = APIRouter(tags=["friends"])


def _handle(call):
    try:
        return call()
    except fs.FriendError as e:
        raise HTTPException(status_code=e.status, detail=e.detail)


@router.get("/users/search")
def search_users(
    q: str = "",
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[dict]:
    return fs.search_users(db, user.id, q)


@router.get("/friends")
def list_friends(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    return fs.list_friends(db, user.id)


@router.post("/friends/request")
def request_friend(
    body: FriendRequestBody,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    status = _handle(lambda: fs.request_friend(db, user.id, body.username))
    return {"status": status}


@router.post("/friends/{request_id}/accept")
def accept_request(
    request_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    status = _handle(lambda: fs.respond_to_request(db, user.id, request_id, accept=True))
    return {"status": status}


@router.post("/friends/{request_id}/decline")
def decline_request(
    request_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    status = _handle(lambda: fs.respond_to_request(db, user.id, request_id, accept=False))
    return {"status": status}


@router.delete("/friends/{friend_id}")
def unfriend(
    friend_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    _handle(lambda: fs.unfriend(db, user.id, friend_id))
    return {"ok": True}
