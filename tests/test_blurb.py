"""Blend blurb generator (pure, no model load)."""

from __future__ import annotations

from app.services.blurb_service import blend_blurb
from app.services.copy import join_list

GENRES = ["Action", "Crime", "Drama", "Sci-Fi", "Comedy"]


def test_join_grammar():
    assert join_list([]) == "a bit of everything"
    assert join_list(["a"]) == "a"
    assert join_list(["a", "b"]) == "a and b"
    assert join_list(["a", "b", "c"]) == "a, b and c"


def test_blurb_names_person_and_top_genres():
    vec = [0.1, 0.9, 0.8, 0.2, 0.0]  # Crime, Drama top
    blurb = blend_blurb("Sara", vec, GENRES, overlap_count=7)
    assert "Sara" in blurb
    assert "crime" in blurb and "drama" in blurb
    assert "gravitate" in blurb


def test_blurb_positive_overlap_uses_taste_summary():
    vec = [0.5, 0.4, 0.3, 0.2, 0.1]
    blurb = blend_blurb("Sara", vec, GENRES, overlap_count=1)
    assert "both gravitate to" in blurb


def test_blurb_no_overlap_message():
    vec = [0.5, 0.4, 0.3, 0.2, 0.1]
    blurb = blend_blurb("Sara", vec, GENRES, overlap_count=0)
    assert "different tastes" in blurb
