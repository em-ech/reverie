# Reverie — Architecture Audit

Adversarial pre-build review of `ARCHITECTURE.md` + `PROJECT_PLAN.md`, run across four
independent lenses (data/leakage, model/training, evaluation, scope/integration) before
any code is written. Findings are de-duplicated and ranked. Mitigations are folded into
the revised `ARCHITECTURE.md`; experiments to resolve open questions are in `EXPERIMENTS.md`.

Severity: **C** critical (fix before coding) · **H** high · **M** medium · **L** low.

> **Post-submission update (2026-06).** This audit reviewed the frozen GRU build and
> remains the record of that review. One scope limitation it implied has since been
> resolved out of band: the GRU trains on MovieLens `ml-1m` (films up to the year 2000)
> and cannot recommend modern titles or score a brand-new user. After the deadline we
> added a **neural collaborative-filtering** model (~9.9M Letterboxd ratings) that does
> both; it beats every naive baseline globally and for 94% of users. See
> `notebooks/ncf_collaborative.ipynb` and the update banner in `ARCHITECTURE.md`.

---

## Critical — fix before writing code

### C1. Training-window scheme is contradictory and load-bearing

Leave-one-out "the rest → train" implies ~610 training sequences (one per user); the plan
also says "sliding windows," implying ~98K. This single choice decides whether a
3k–10k-way softmax is learnable at all.
**Fix:** **all-prefix / next-item-at-every-step for TRAINING** (each user contributes
~len−1 teacher-forced targets); **leave-one-out only for EVAL** (last item = test,
second-to-last = val). Documented in revised §2.

### C2. Global statistics leak the test set

`MIN_COUNT` filtering, popularity counts, and rating normalization are specified over the
full dataset → test interactions influence the vocabulary, the baseline, and the scaler.
**Fix:** compute `MIN_COUNT` and `popularity` on **train interactions only**; use the
fixed rating scale `(r-0.5)/4.5` (no fitting); held-out targets that fall out of catalog
count as automatic misses (not silently re-included).

### C3. Padding index collides with a real movie

`mask_zero` treats integer 0 as PAD, but "contiguous indices" can assign a real movie to 0.
**Fix:** reserve index 0 for PAD, real movie ids start at 1, softmax dim = `V+1`, exclude
index 0 from the eval candidate set and recommendations, assert `0 not in targets`. Also
zero the genre/rating channels on padded steps and confirm the mask propagates through
`Concatenate` to the GRU.

---

## Existential — the quantitative thesis may not hold

### H1. The RNN may not beat the popularity baseline

610 users / multi-year MovieLens "sessions" is a weak regime for sequential signal; SCCE
under class imbalance collapses toward popularity. The project stakes its thesis on
"RNN beats baselines," and Phase 4 (day 6–8) is too late to discover failure.
**Fix:** (a) make **`ml-1m` the primary corpus** from day 1, `ml-latest-small` the
fallback; (b) all-prefix augmentation (C1); (c) **logit adjustment / class-balanced loss**
to counter popularity collapse; (d) **de-risk in days 2–3** — bare GRU vs most-popular on
ml-1m before building anything else; (e) **pre-register** the success criterion AND a
fallback narrative: if the RNN only ties popularity, the finding becomes "sequential
signal is weak on sparse multi-year histories, with rigorous evidence" — legitimate and
defensible. Decide now, not in week 2.

### H2. The softmax-derived taste vector collapses to the genre marginal

`taste = Σ P(movie)·genre(movie)` is a valid expectation but, under a near-popularity
softmax, ≈ the global genre marginal for everyone → TV cosine becomes "recommend generic
genres to all," and the taste-drift demo may not visibly move.
**Fix:** **mean-center** the genre space (subtract the global marginal from taste and
candidate vectors before cosine); compute taste from the **top-K** of the softmax (e.g.
top 50–100), not the full tail; report **across-user taste-vector variance** and softmax
entropy as diagnostics. If variance ≈ 0, the feature is dead — say so.

---

## High

### H3. `recurrent_dropout` silently disables the cuDNN fast path

Sets training 5–20× slower; on Apple Silicon RNN kernels are also fragile.
**Fix:** drop `recurrent_dropout`; use input `dropout` + `Dropout(0.3)` before the head +
L2 + early stopping. Day-1 spike: confirm the GRU trains correctly/fast on the demo machine.

### H4. Early stopping on val NDCG@10 is a custom callback, not a flag

NDCG@10 is a full-catalog ranking metric; monitoring it requires per-epoch ranking over
all val users.
**Fix:** **Phase 2** early-stop on **val loss** (free, correct enough to lock the
pipeline); **Phase 3** add an NDCG callback that reuses the _same_ ranking function as the
final evaluator (DRY). Verify val-loss and val-NDCG correlate before trusting NDCG.

### H5. Baselines are weak by default; item-kNN must be a required comparison

most-popular and most-recent-genre are (weakly) non-personalized; a tuned **item-kNN**
often beats shallow RNNs on small data and is the baseline a knowledgeable grader expects.
**Decision (resolves a reviewer disagreement):** **keep item-kNN as a required baseline** —
~15 lines of scikit-learn, already in the stack. If the RNN can't beat it, that is the
project's most important finding.

### H6. No variance or significance treatment

Single point estimates over ~610 users, single seed → "beats baseline" may be noise or a
lucky seed.
**Fix:** bootstrap 95% CIs over users (resample 1000×); **paired Wilcoxon signed-rank** on
per-user reciprocal ranks (RNN vs baseline); train the **frozen config under ≥3 seeds**,
report mean ± std. Test set opened **exactly once** on the val-frozen config.

### H7. Documents disagree on the model (dual-head vs derived vector)

`PROJECT_PLAN.md` still describes a trained "Head B"; `ARCHITECTURE.md` uses a derived
taste vector. The 25%-weighted component has two sources of truth.
**Fix:** the **derived-vector design wins** (no unsupervised head). Reconcile the plan.

---

## Medium

### M1. TV feedback loop is conceptually broken for out-of-catalog items

A TV show has no movie embedding to feed the RNN input `[emb ⊕ genre ⊕ rating]`.
**Fix:** demonstrate the feedback re-rank on **movie** rejections (in-catalog, correct
behavior); for TV rejections, **update the derived taste vector directly** (subtract the
item's genre vector) rather than running it through the RNN.

### M2. Seen-item filtering unspecified → rigged comparison risk

Whether already-watched items are removed from the candidate set changes every metric and
must be identical across all models.
**Fix:** exclude each user's train+val history from the rankable set for **all** models
uniformly; evaluate on the **raw** scoring ranking (diversity/de-dup re-rank only in serving).

### M3. α-blend over-engineered for in-catalog movies

Blending genre cosine into the softmax score for items the RNN saw mostly re-injects the
popularity-genre marginal.
**Fix:** use **α≈1** (pure softmax) for in-catalog movies; the cosine bridge earns its
keep for out-of-catalog TV only. Drop the α sweep (or keep one quick check).

### M4. TV genre cosine over ~18 coarse genres demos poorly

Many shows share near-identical genre vectors → "you like Drama, here are 10 dramas."
**Fix:** enrich the TV vector with a cheap **TF-IDF over `imdb_tvshows` descriptions**
(~1 hr) OR curate demo personas where genre-cosine looks sensible and state the coarseness
as a known limitation. Log genre-mapping coverage; target ≥85%.

### M5. "Qualitative-only" TV is a cop-out — build a small quantitative proxy

You have held-out personal Netflix TV interactions.
**Fix (resolves a reviewer disagreement):** leave-one-out over the ~15–30 resolved personal
TV shows, content-cosine rank vs the IMDb catalog (3,000), report HR@k / MRR with explicit
"N small, single user, illustrative — vs random = k/3000," plus a **shuffle-genre sanity
floor** showing HR collapses to random. Keep "qualitative" as the primary framing, backed
by one honest number.

### M6. Environment drift across 5 laptops + Streamlit/TF demo

TF is the worst cross-machine dependency; full `.keras` files break across TF versions.
**Fix:** pin exact `tensorflow==X.Y.Z`; one Python version; designate **one training
machine + one demo machine**; save **weights + architecture-in-code**, not full-model
serialization; `@st.cache_resource` + **warm the model on startup**; recorded backup demo
as the real fallback. The frontend owner consumes artifacts, doesn't train.

### M7. Short-history inference (1–2 items) unspecified

The personal-import path can feed ultra-short or empty histories.
**Fix:** define behavior: empty → genre-seeded/popular fallback; 1–2 items → pad+predict
or route to fallback; document the minimum trained history length.

---

## Low / process

- **L1. Deterministic ordering:** sort by `(timestamp, movieId)`, stable sort, fixed seeds,
  assert no duplicate `(user, movie)` — so all 5 machines produce identical splits.
- **L2. Notebook/`src` drift:** notebooks import `src/` functions; `src/gru_model/recommend.py` is the
  only inference path.
- **L3. Letterboxd is unordered:** keep strictly in the qualitative demo; never in a metric.
- **L4. Sampled-negative eval (E7):** run once as a methodology slide (full-ranking is the
  headline); cite Krichene & Rendle (KDD 2020). Cheap honesty differentiator — keep.

---

## What's already fine (don't change)

- Full-catalog ranked eval over sampled negatives — correct at this scale.
- Leave-one-out leakage note on input features (features at `t` describe the past) — correct.
- Derived taste vector replacing the dual-head — a sound simplification.
- Early stopping aligned to a ranking metric (once H4's mechanics are handled).

---

## Over-engineering to cut (protect the core 50% = model + integration)

TMDb API (→ hand-built title CSV) · taste-drift animation (→ static before/after radar) ·
serendipity dial (→ stretch) · binge simulator (stretch) · persistent learning (stretch) ·
the 8-axis experiment sweep (→ the focused set in `EXPERIMENTS.md`).

---

## The one thing to do first

Resolve C1 + C2 + H1 together **before any code**: lock all-prefix-train / LOO-eval, move
every count-based statistic behind the train split, and de-risk "does the RNN beat
popularity on ml-1m" in the first three days. Everything downstream depends on these.
