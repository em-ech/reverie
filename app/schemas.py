"""Pydantic request/response models. ORM objects never cross the HTTP boundary;
password hashes are never serialized."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=30, pattern=r"^[A-Za-z0-9_]+$")
    password: str = Field(min_length=6, max_length=128)
    display_name: str | None = Field(default=None, max_length=60)


class LoginRequest(BaseModel):
    username: str
    password: str


class UserOut(BaseModel):
    id: int
    username: str
    display_name: str | None = None


class AuthResponse(BaseModel):
    token: str
    user: UserOut


class MeResponse(UserOut):
    history_count: int
    friend_count: int


class HistoryAddRequest(BaseModel):
    movieId: int
    rating: float = Field(default=4.0, ge=0.5, le=5.0)


class HistoryRatingRequest(BaseModel):
    rating: float = Field(ge=0.5, le=5.0)


class FriendRequestBody(BaseModel):
    username: str
