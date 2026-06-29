"""Friend-graph logic: relationships, requests, and listings.

A friendship is one row per ordered (requester, addressee) pair; the two users
are friends when that row is 'accepted', in either direction. Reused by the
friends router and (later) the blend router's friendship check.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session

from app.models import Friendship, HistoryItem, User

# Relationship labels surfaced to the UI.
SELF = "self"
NONE = "none"
FRIENDS = "friends"
PENDING_OUT = "pending_out"  # I requested them
PENDING_IN = "pending_in"    # they requested me


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _pair_row(db: Session, a: int, b: int) -> Friendship | None:
    """Any friendship row between a and b, in either direction."""
    return db.scalar(
        select(Friendship).where(
            or_(
                and_(Friendship.requester_id == a, Friendship.addressee_id == b),
                and_(Friendship.requester_id == b, Friendship.addressee_id == a),
            )
        )
    )


def are_friends(db: Session, a: int, b: int) -> bool:
    row = _pair_row(db, a, b)
    return row is not None and row.status == "accepted"


def relationship(db: Session, me: int, other: int) -> str:
    if me == other:
        return SELF
    row = _pair_row(db, me, other)
    if row is None or row.status in ("declined", "blocked"):
        return NONE
    if row.status == "accepted":
        return FRIENDS
    # pending
    return PENDING_OUT if row.requester_id == me else PENDING_IN


def _history_count(db: Session, user_id: int) -> int:
    return db.scalar(
        select(func.count()).select_from(HistoryItem).where(HistoryItem.user_id == user_id)
    ) or 0


def search_users(db: Session, me: int, q: str, limit: int = 10) -> list[dict]:
    """Username search (case-insensitive substring), excluding self, with the
    viewer's relationship to each result."""
    ql = q.lower().strip()
    if not ql:
        return []
    users = db.scalars(
        select(User)
        .where(User.username_ci.contains(ql), User.id != me)
        .order_by(User.username_ci)
        .limit(limit)
    ).all()
    return [
        {
            "id": u.id,
            "username": u.username,
            "display_name": u.display_name,
            "relationship": relationship(db, me, u.id),
        }
        for u in users
    ]


class FriendError(Exception):
    """Raised with an HTTP status + message for the router to surface."""

    def __init__(self, status: int, detail: str):
        self.status = status
        self.detail = detail
        super().__init__(detail)


def request_friend(db: Session, me: int, username: str) -> str:
    target = db.scalar(select(User).where(User.username_ci == username.lower()))
    if target is None:
        raise FriendError(404, "No user with that username")
    if target.id == me:
        raise FriendError(400, "You can't friend yourself")

    row = _pair_row(db, me, target.id)
    if row is None:
        db.add(Friendship(requester_id=me, addressee_id=target.id, status="pending"))
        db.commit()
        return PENDING_OUT
    if row.status == "accepted":
        raise FriendError(409, "You're already friends")
    if row.status == "pending":
        if row.requester_id == me:
            raise FriendError(409, "Request already sent")
        # They already requested me -> mutual intent, accept it.
        row.status = "accepted"
        row.responded_at = _now()
        db.commit()
        return FRIENDS
    # declined/blocked -> re-open as a fresh request from me
    row.requester_id, row.addressee_id = me, target.id
    row.status = "pending"
    row.responded_at = None
    db.commit()
    return PENDING_OUT


def respond_to_request(db: Session, me: int, request_id: int, accept: bool) -> str:
    fr = db.get(Friendship, request_id)
    if fr is None or fr.addressee_id != me or fr.status != "pending":
        raise FriendError(404, "No pending request")
    fr.status = "accepted" if accept else "declined"
    fr.responded_at = _now()
    db.commit()
    return fr.status


def unfriend(db: Session, me: int, other_id: int) -> None:
    row = _pair_row(db, me, other_id)
    if row is None or row.status != "accepted":
        raise FriendError(404, "Not friends")
    db.delete(row)
    db.commit()


def _user_brief(db: Session, u: User, *, with_count: bool = False) -> dict:
    out = {"id": u.id, "username": u.username, "display_name": u.display_name}
    if with_count:
        out["history_count"] = _history_count(db, u.id)
    return out


def list_friends(db: Session, me: int) -> dict:
    """Accepted friends + incoming and outgoing pending requests."""
    accepted = db.scalars(
        select(Friendship).where(
            Friendship.status == "accepted",
            or_(Friendship.requester_id == me, Friendship.addressee_id == me),
        )
    ).all()
    friends = []
    for fr in accepted:
        other_id = fr.addressee_id if fr.requester_id == me else fr.requester_id
        other = db.get(User, other_id)
        if other:
            friends.append(_user_brief(db, other, with_count=True))

    incoming = []
    for fr in db.scalars(
        select(Friendship).where(
            Friendship.addressee_id == me, Friendship.status == "pending"
        )
    ).all():
        u = db.get(User, fr.requester_id)
        if u:
            incoming.append({"requestId": fr.id, **_user_brief(db, u)})

    outgoing = []
    for fr in db.scalars(
        select(Friendship).where(
            Friendship.requester_id == me, Friendship.status == "pending"
        )
    ).all():
        u = db.get(User, fr.addressee_id)
        if u:
            outgoing.append({"requestId": fr.id, **_user_brief(db, u)})

    return {"friends": friends, "incoming": incoming, "outgoing": outgoing}
