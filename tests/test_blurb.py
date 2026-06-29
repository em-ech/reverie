"""Blend blurb generator (pure, no model load)."""

from __future__ import annotations

from app.services.blurb_service import _join, blend_blurb

GENRES = ["Action", "Crime", "Drama", "Sci-Fi", "Comedy"]


def test_join_grammar():
    assert _join([]) == "a bit of everything"
    assert _join(["a"]) == "a"
    assert _join(["a", "b"]) == "a and b"
    assert _join(["a", "b", "c"]) == "a, b and c"


def test_blurb_names_top_genres_and_overlap():
    vec = [0.1, 0.9, 0.8, 0.2, 0.0]  # Crime, Drama top
    blurb = blend_blurb("Sara", vec, GENRES, overlap_count=7)
    assert "Sara" in blurb
    assert "crime" in blurb and "drama" in blurb
    assert "7 films" in blurb


def test_blurb_singular_film():
    vec = [0.5, 0.4, 0.3, 0.2, 0.1]
    assert "1 film lights" in blend_blurb("Sara", vec, GENRES, overlap_count=1)


def test_blurb_no_overlap_message():
    vec = [0.5, 0.4, 0.3, 0.2, 0.1]
    blurb = blend_blurb("Sara", vec, GENRES, overlap_count=0)
    assert "different tastes" in blurb
    assert "middle ground" in blurb
