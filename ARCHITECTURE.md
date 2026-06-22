# Reverie — Architecture Design (v2, post-audit)

Concrete design for the model, data pipeline, recommendation logic, evaluation, and
serving. Revised to fold in every mitigation from `AUDIT.md`. Audit tags like `[C1]`
reference findings there. Numbers are starting points confirmed by `EXPERIMENTS.md`.

---

## 1. End-to-end data flow

```
MovieLens ratings ─┐
                   ├─▶ preprocess ─▶ all-prefix train windows ─▶ train RNN ─▶ weights + arch-in-code + encoders
genres ────────────┘                                                              │
                                                                                  ▼
imdb_tvshows.csv ─▶ shared-genre catalog (+TF-IDF) ────────────────────▶ recommend(history)
Letterboxd / Netflix export ─▶ map to catalog ─▶ history ───────────────▶        │
                                                                                  ▼
                                                                      React dashboard (FastAPI)
```

---

## 2. Data pipeline (training corpus: MovieLens)

- **Primary dataset: `ml-1m`** (6,040 users, ~3,700 movies, 1M ratings). `ml-latest-small`
  is the fallback only. `[H1]` — denser catalog and ~10× sequences make the softmax learnable.
- **Deterministic ordering `[L1]`:** sort each user's interactions by `(timestamp, movieId)`
  with a stable sort; fixed seeds; assert no duplicate `(user, movie)`. All machines must
  produce identical splits.
- **Split first, then compute statistics `[C2]`:** per user, leave-one-out — last
  interaction = **test**, second-to-last = **val**, the rest = **train pool**.
- **Train-only statistics `[C2]`:** `MIN_COUNT` item filtering (start 5; sweep) and
  `popularity` counts are computed on the **train pool only**. A held-out target whose item
  is below threshold counts as an automatic miss at eval (never silently re-added).
- **Training windows — all-prefix / next-item-at-every-step `[C1]`:** within each user's
  train pool, every prefix predicts the next item: `(i_1..i_t) → i_{t+1}` for all valid `t`.
  This yields tens of thousands of teacher-forced targets, not one per user. Leave-one-out
  is used **only for evaluation**, never to generate training windows.
- **Eval histories `[C2]`:** val target = `n-1`, history `1..n-2`; test target = `n`,
  history `1..n-1` (the val item **is** included in the test history). Assert
  `len(test_history) == len(val_history) + 1`.
- **Windowing/padding `[C3]`:** max length `L=50` (sweep). **Index 0 reserved for PAD**;
  real movie ids start at 1; softmax output dim = `V+1`; pre-pad / left-truncate; genre and
  rating channels are **zeroed on padded steps**; confirm the mask propagates through
  `Concatenate` to the GRU (feed an all-pad sequence; the GRU output must be invariant).
- **Per-timestep input features (concatenated):**
  - movie embedding, dim `d_movie=32` start (small first — sweep up only if val improves) `[M2-cap]`
  - movie genre vector in the **shared genre vocabulary** (multi-hot)
  - **fixed-scale** rating `(r-0.5)/4.5` — no fitted scaler `[C2]`
  - _Leakage note:_ features at step `t` describe the item watched at `t` (past); the target
    `i_{t+1}` (id only) is never an input. Unit-assert no target-position feature row leaks in.

---

## 3. Model

```
input seq (L, feat) ─▶ Masking ─▶ GRU(128, dropout=0.2) ─▶ Dropout(0.3) ─▶ Dense(V+1, softmax)
```

- **Cell:** GRU default; LSTM is one ablation. 1 layer first; 2 only if it helps.
- **Regularization `[H3]`:** **no `recurrent_dropout`** (it disables the cuDNN fast path and
  is fragile on Apple Silicon). Use GRU input `dropout=0.2` + `Dropout(0.3)` before the head
  - L2 on embeddings + early stopping.
- **Objective:** sparse categorical cross-entropy on the next item, Adam (lr 1e-3), batch 128.
- **Popularity-bias correction `[H1]`:** apply **logit adjustment** (subtract the log
  train-prior from logits) / class-balanced loss to stop the softmax collapsing to popularity.
  This is the principled fix; the serendipity penalty is only a post-hoc cosmetic.
- **Model selection `[H4]`:** Phase 2 early-stop on **val loss**; Phase 3 swap in a custom
  callback for **val NDCG@10** that reuses the evaluator's ranking function. Verify the two
  correlate before trusting NDCG.

### Taste vector — derived, mean-centered, top-K `[H2, H7]`

Single training objective (next-item softmax). The taste profile is a deterministic
function of the softmax — no second trained head:

```
P_topK     = renormalized softmax over its top-K items (K≈50–100)
taste_raw  = Σ_{i∈topK} P_topK(i) · genre(i)
taste      = taste_raw − genre_marginal        # mean-centered: deviation from average
```

Candidate genre vectors are mean-centered the same way before cosine, so similarity
measures _deviation from the average viewer_, not "everyone likes Drama." Diagnostics to
report: softmax entropy and across-user variance of `taste` (if ≈0, the signal is dead).

---

## 4. Recommendation logic (movies + TV)

### Shared genre vocabulary `[M4]`

Hand-author an `{imdb_genre_string → movielens_genre}` map to the 18-genre MovieLens vocab.
Log coverage ("% of TV catalog with ≥1 mapped genre"); treat <85% as a bug. Optionally
enrich TV item vectors with a **TF-IDF over `imdb_tvshows` descriptions** for separation.

### Scoring `[M3]`

- **In-catalog movies:** `score = P_headA(movie)` (α≈1 — pure softmax; the RNN saw these).
- **Out-of-catalog TV / titles:** `score = cosine(taste, genre_centered(item))`.
- **Seen-item filtering `[M2]`:** exclude each user's train+val history from the rankable
  set for **all** models, uniformly.
- Top-N after de-dup + a genre-diversity pass — **serving path only**, never in eval.

### Cold / short history `[M7]`

Empty → genre-seeded or most-popular fallback. 1–2 items → pad+predict or fallback. Document
the minimum trained history length so the demo never feeds out-of-distribution sequences.

---

## 5. Learning from mistakes (in-session) `[M1]`

- **Movie rejection (primary demo path):** append a synthetic step `(movie, low rating)` to
  the live history → re-run `recommend(history)` → corrected top-N. In-catalog, behaves correctly.
- **TV rejection:** the item has no movie embedding, so do **not** feed it through the RNN —
  **update the derived taste vector directly** (subtract its centered genre vector) and re-score.
- No retraining. Before the demo, script 5 reject-and-rerank steps and measure top-10 churn;
  if churn ≈ 0, add an explicit hard penalty on the rejected item + its genre neighbors.

---

## 6. Evaluation `[H5, H6, M2, M5]`

- **Protocol:** leave-one-out; rank the held-out true item against the **full** catalog
  (index 0/PAD excluded). Sampled-negative eval run **once** as a methodology slide (cite
  Krichene & Rendle, KDD 2020); full-ranking is the headline.
- **Metrics — reported honestly `[H1-eval]`:** under one positive/user, **Recall@10 ≡ HR@10**,
  and NDCG@10 / MRR are rank summaries of the same per-user rank. Report **HR@10 + MRR** (state
  the relationship), plus top-1 as a sanity check. Don't dress one signal up as four metrics.
- **Baselines (all required):** (a) most-popular (train-only counts), (b) most-recent-genre
  popularity (defined: popularity within the genre-set of the user's last-K _train_ items,
  seen items excluded), (c) **item-kNN** (cosine on the user-item matrix, scikit-learn). The
  RNN must beat all three to justify the architecture.
- **Significance `[H6]`:** bootstrap 95% CIs over users (1000×); paired Wilcoxon signed-rank
  on per-user reciprocal ranks (RNN vs each baseline); final config trained under **≥3 seeds**,
  report mean ± std. **Test set opened exactly once** on the val-frozen config.
- **Overfitting evidence:** train vs val curves; monitor the gap.
- **TV quantitative proxy `[M5]`:** leave-one-out over the ~15–30 resolved personal Netflix
  TV shows, content-cosine rank vs the 3,000-show catalog; report HR@k / MRR with "N small,
  single user, illustrative — vs random = k/3000," plus a shuffle-genre sanity floor.

---

## 7. Serving (FastAPI + React) `[M6]`

- Pin exact `tensorflow==X.Y.Z`; one Python version across the team.
- Save **weights + architecture-in-code** (not full-model `.keras`) for version robustness;
  provide a "rebuild artifacts on this machine" path.
- FastAPI loads model + encoders + catalog once at startup and **warms it with a dummy
  `recommend([])`** so cold start never hits the grader. React frontend calls the API.
- Designate **one training machine + one demo machine**; the frontend owner consumes
  artifacts, never trains. Recorded backup demo is the real fallback, not a checkbox.
- `recommend(history)` (in `src/recommend.py`) is the **only** inference path; notebooks
  import it, never reimplement it `[L2]`. No precomputed/hardcoded predictions.
- Artifacts contract: `weights.h5`, `model_config.json`, `movie_index.json`,
  `genre_matrix.npy`, `genre_marginal.npy`, `tv_catalog.parquet`, `popularity_train.npy`.

---

## 8. Pre-registered success criterion & fallback `[H1]`

- **Success:** on `ml-1m`, the RNN beats most-popular, most-recent-genre, AND item-kNN on
  HR@10 / MRR with a bootstrap CI on the difference that excludes 0.
- **Day 2–3 gate:** bare GRU vs most-popular on `ml-1m`. If it does not beat popularity by
  day 3, escalate (more augmentation, smaller V, logit adjustment) while there is runway.
- **Fallback narrative (decided now):** if the RNN only ties the baselines, the project
  reports "sequential signal is weak on sparse multi-year watch histories — here is the
  rigorous evidence (CIs, ablations, the item-kNN comparison)." This is a legitimate,
  presentable result, not a failure.
