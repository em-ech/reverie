"""Friend-graph logic tests (in-memory SQLite, no model load)."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db import Base
from app.models import User
from app.services import friend_service as fs


@pytest.fixture()
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def _user(db: Session, name: str) -> User:
    u = User(username=name, username_ci=name.lower(), password_hash="x")
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


def test_request_creates_pending_both_directions(db):
    a, b = _user(db, "alice"), _user(db, "bob")
    assert fs.request_friend(db, a.id, "bob") == fs.PENDING_OUT
    assert fs.relationship(db, a.id, b.id) == fs.PENDING_OUT
    assert fs.relationship(db, b.id, a.id) == fs.PENDING_IN
    assert not fs.are_friends(db, a.id, b.id)


def test_accept_makes_friends(db):
    a, b = _user(db, "alice"), _user(db, "bob")
    fs.request_friend(db, a.id, "bob")
    inbox = fs.list_friends(db, b.id)["incoming"]
    assert len(inbox) == 1
    fs.respond_to_request(db, b.id, inbox[0]["requestId"], accept=True)
    assert fs.are_friends(db, a.id, b.id)
    assert fs.relationship(db, a.id, b.id) == fs.FRIENDS


def test_mutual_request_auto_accepts(db):
    a, b = _user(db, "alice"), _user(db, "bob")
    fs.request_friend(db, a.id, "bob")
    # bob requesting alice back should accept the existing request
    assert fs.request_friend(db, b.id, "alice") == fs.FRIENDS
    assert fs.are_friends(db, a.id, b.id)


def test_duplicate_request_conflicts(db):
    a, _ = _user(db, "alice"), _user(db, "bob")
    fs.request_friend(db, a.id, "bob")
    with pytest.raises(fs.FriendError) as exc:
        fs.request_friend(db, a.id, "bob")
    assert exc.value.status == 409


def test_cannot_friend_self_or_unknown(db):
    a = _user(db, "alice")
    with pytest.raises(fs.FriendError) as e1:
        fs.request_friend(db, a.id, "alice")
    assert e1.value.status == 400
    with pytest.raises(fs.FriendError) as e2:
        fs.request_friend(db, a.id, "ghost")
    assert e2.value.status == 404


def test_unfriend_and_relist(db):
    a, b = _user(db, "alice"), _user(db, "bob")
    fs.request_friend(db, a.id, "bob")
    inbox = fs.list_friends(db, b.id)["incoming"]
    fs.respond_to_request(db, b.id, inbox[0]["requestId"], accept=True)
    fs.unfriend(db, a.id, b.id)
    assert not fs.are_friends(db, a.id, b.id)
    assert fs.list_friends(db, a.id)["friends"] == []


def test_search_excludes_self_and_labels_relationship(db):
    a, b = _user(db, "alice"), _user(db, "bob")
    fs.request_friend(db, a.id, "bob")
    results = fs.search_users(db, a.id, "b")
    assert [r["username"] for r in results] == ["bob"]
    assert results[0]["relationship"] == fs.PENDING_OUT
    assert fs.search_users(db, a.id, "alice") == []  # self excluded
