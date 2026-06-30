"""Tiny template-based blurb generator for the Blend (no LLM needed)."""

from __future__ import annotations

from app.services.copy import genre_label, join_list


def blend_blurb(
    friend_name: str,
    blend_vec: list[float],
    genre_names: list[str],
    overlap_count: int,
) -> str:
    top = [
        genre_label(g)
        for g, _ in sorted(zip(genre_names, blend_vec), key=lambda x: x[1], reverse=True)[:3]
    ]
    genres = join_list(top)
    if overlap_count == 0:
        return f"You and {friend_name} have pretty different tastes."
    return f"You and {friend_name} both gravitate to {genres}."
