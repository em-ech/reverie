# Deep Learning Final Project — Plan

**Course:** Deep Learning (IE University)
**Group:** 5 members
**Deliverable:** Functional MVP (GitHub repo) + 15-min pitch with live demo (PDF deck)
**Scope reality:** ~2 weeks, trained on local machines → lean MVP. Train on MovieLens
(**`ml-1m` primary** after the audit; `ml-latest-small` fallback), recommend TV + movies
via a content bridge, demo on real personal data. Tuning kept shallow; stretch goals optional.

**Companion docs:** [`ARCHITECTURE.md`](ARCHITECTURE.md) (authoritative model/data/serving
spec), [`AUDIT.md`](AUDIT.md) (pre-build adversarial review + mitigations),
[`EXPERIMENTS.md`](EXPERIMENTS.md) (runnable experiment protocol).

**Post-audit cut list (protect the core 50% = model + integration):** TMDb API → hand-built
title CSV; taste-drift animation → static before/after radar; serendipity dial → stretch;
persistent learning → stretch; the 8-axis sweep → the focused matrix in `EXPERIMENTS.md`.
Feedback loop is demoed on **movies** (TV rejection updates the taste vector directly).

---

## 1. The product

**Reverie** — a sequential recommender. It reads a viewer's watch history, learns
their evolving taste, and recommends what to watch next across **both movies and TV
shows**, served through an interactive React dashboard (model served by a FastAPI API). The system also **learns
from being wrong**: when the user rejects a recommendation, it corrects in real time.

### Business case (rubric: 20%)

- **Problem:** streaming catalogs are huge; users churn when they cannot find
  something to watch. Recommendation drives the majority of viewing on platforms
  like Netflix.
- **Value proposition:** a personalized "what to watch next" engine increases
  watch-time and session length, reduces decision fatigue, and lowers churn.
- **Quantify in the deck:** tie to retention / watch-time uplift and the cost of
  churn. Frame as engagement-per-user and reduced content-discovery time.

---

## 2. Data inventory and roles

| Dataset                                               | Role                                 | Notes                                                                                                     |
| ----------------------------------------------------- | ------------------------------------ | --------------------------------------------------------------------------------------------------------- |
| MovieLens `ml-latest-small`                           | **Training corpus**                  | ~100K ratings, 610 users, ~9,700 movies; timestamps + genres + tags. The only real watch-_sequence_ data. |
| Letterboxd export (`ratings.csv`, `watched.csv`)      | **Personal movie demo**              | 459 rated films, 230 map exactly to MovieLens; real star ratings. Rich real taste profile.                |
| Netflix export (`ViewingActivity.csv`, `Ratings.csv`) | **Personal TV demo + real feedback** | ~1,820 rows, ~81% TV, Spanish-localized; real thumbs. Curated subset only.                                |
| `imdb_tvshows.csv`                                    | **TV candidate catalog**             | 3,000 shows with genres + descriptions; English titles.                                                   |
| Netflix Prize (`combined_data_*`)                     | **Set aside**                        | No genres, 2GB+, day-level dates. Optional "this scales to 100M ratings" deck talking point only.         |
| IMDB reviews (`IMDB Dataset.csv`)                     | **Set aside**                        | 50K-review sentiment dataset; wrong task, no users/titles/sequences.                                      |

### Data caveats to state openly (rubric rewards honesty)

- **Letterboxd has no reliable chronology** (bulk-logged, 16 distinct dates, empty
  diary). It is a rated-film _set_, not an ordered sequence. The model learns order
  from MovieLens; Letterboxd supplies the real taste profile for inference.
- **Netflix TV is Spanish-localized** → curated manual/TMDb resolution of ~20-30 top
  shows to English/genres for the demo. Coverage is partial.
- **TV recommendations are qualitative** — there is no TV interaction dataset to
  train or score against. Quantitative eval lives on the movie side.
- **Privacy:** personal exports also contain IPs, billing, account, chat data. Only
  touch viewing/ratings files; **gitignore all personal data**; never commit it.

---

## 3. Technical design (rubric: 25% model)

### Architecture: RNN (GRU/LSTM), justified

Watch order carries signal — recently watched titles predict the next watch better
than a static average profile. That sequential dependency is exactly what an RNN
models (standard session-based / sequential recommendation, GRU4Rec family).

### Preprocessing (MovieLens, the training corpus)

- Encode `movieId` → contiguous indices; keep a reverse map for the frontend.
- Filter to movies with a minimum rating count (e.g. ≥5–10) to shrink the wide,
  sparse catalog and reduce popularity bias.
- Per user: sort interactions by `timestamp` → one chronological sequence.
- Use **all interactions** (not filtered by rating).
- Sliding windows of length `L` (start L≈50, tune): input = movies `1..n-1`,
  target = movie `n`. Pad/truncate with masking.
- Per-timestep input features (hybrid): movie embedding (~50-d) ⊕ genre
  representation (multi-hot/embedding) ⊕ **rating value** (normalized).
- Split (leave-one-out, standard for sequential rec): per user, last item = test,
  second-to-last = validation, the rest = train.

### Model — single head + derived taste vector

> Superseded the earlier "dual head" after the audit. The authoritative model spec is
> [`ARCHITECTURE.md`](ARCHITECTURE.md); this is the summary.

```
sequence ─▶ [movie emb ⊕ genre ⊕ rating] ─▶ GRU ─▶ Dropout ─▶ Dense softmax (next movie)
                                                       │
                                                       └─▶ taste vector = Σ P(movie)·genre(movie)  (DERIVED, not trained)
```

- **Softmax head** → quantitative evaluation and the movie recommender.
- **Derived taste vector** (mean-centered, top-K) → scores _any_ candidate title (movie or
  TV from `imdb_tvshows.csv`) by genre cosine → enables **TV + movie** recommendations and
  fixes title-mapping (match on content, not exact title). No separately-trained head.
- Overfitting prevention: dropout (no `recurrent_dropout` — keeps the cuDNN path), early
  stopping (val loss → val NDCG@10 in Phase 3), L2 on embeddings, masking. Train/val curves
  as evidence.

### Evaluation

> Full protocol, baselines, and significance testing in [`EXPERIMENTS.md`](EXPERIMENTS.md).

- Metrics: **HR@10 + MRR** (under leave-one-out, Recall@10 ≡ HR@10; report the relationship
  openly), top-1 as a sanity check.
- **Baselines to beat (all required):** most-popular, most-recent-genre, **and item-kNN**.
- Bootstrap CIs + paired Wilcoxon over users, ≥3 seeds, test opened once.
- TV side: qualitative demo **plus** a small quantitative content-bridge proxy on the
  personal Netflix TV history (illustrative, N small).

---

## 3b. Learning from mistakes (the feedback loop)

Requirement: the system should learn how it went wrong, not just predict once.

### Approach (achievable in 2 weeks, local): in-session correction

- Each recommendation card has reactions: **watched & liked / watched & disliked /
  not interested**.
- A reaction is appended to the user's _live_ sequence as a new step with a rating
  (disliked → low, liked → high). "Not interested" is a soft negative.
- The RNN immediately re-runs inference on the updated sequence → corrected top-N.
  The model steers away from what it got wrong in real time, no retraining.
- Seeded with **real feedback**: Netflix thumbs + Letterboxd stars.

### Stretch (only if time): persistent learning

- Store feedback across sessions and periodically fine-tune so the _weights_ update.
  Heavier (feedback store + retraining loop); explicitly optional.

---

## 3c. Creative differentiators (the "wow" for the demo)

Pitch hook: **a recommender that spans your whole watch life — movies and shows —
argues back, and visibly learns from being wrong, live.** These mostly visualize or
re-rank existing model output, so they add demo impact without extra training cost.

1. **Bring your own history.** Load a real Letterboxd (movies) + Netflix (TV) profile
   and get real recommendations for a real person — far stronger than a synthetic persona.
2. **Taste-drift visualization.** Render current taste as a genre radar/heatmap from
   the model's output. Reject a recommendation → animate the profile shifting and the
   list re-ranking. The audience _sees_ the model course-correct.
3. **Serendipity dial.** Novelty-vs-accuracy slider; doubles as popularity-bias mitigation.
4. **"Why this?" explanations.** Each card cites the recent watches driving it.
5. **Binge-session simulator (stretch).** Auto-plan a night, reject some, it re-plans.

---

## 4. Frontend + integration (rubric: 50% combined)

### React dashboard (vibe coding allowed/encouraged)

- Build / load a watch history (search + add titles, sample personas, or import the
  real Letterboxd/Netflix profile), with a rating slider per title.
- "Recommend" → top-N cards across movies + TV: title, type, year, genre, score.
- Feedback reactions per card → in-session re-rank.
- Polish: taste-drift viz, serendipity dial, "why this" explanations.

### Integration (must be real-time, not hardcoded)

- Training exports: `model.keras`, movie index maps, genre matrix, TV catalog
  embeddings (to `/artifacts`).
- FastAPI loads artifacts once at startup; `recommend(history) -> top_n` runs live
  inference on each request. No precomputed/hardcoded predictions (rubric penalizes it).

---

## 5. Repository structure

```
GroupProject/
  README.md                 # how to run (required deliverable)
  requirements.txt
  data/                     # MovieLens + ALL personal exports (gitignored)
  notebooks/
    01_eda.ipynb
    02_preprocessing.ipynb
    03_model_training.ipynb
    04_evaluation.ipynb
  src/
    data_prep.py            # MovieLens download, clean, encode, sequence-building
    personal_import.py      # parse Letterboxd + Netflix, map titles to catalog
    tv_catalog.py           # ingest imdb_tvshows.csv, build content embeddings
    model.py                # dual-head architecture
    train.py                # training loop, callbacks, export
    evaluate.py             # ranking metrics + baselines
    recommend.py            # inference: next-movie + content-NN over movies + TV
  app/
    api.py                  # FastAPI service (/health, /catalog, /recommend)
  web/                      # React + Vite + Tailwind + shadcn/ui frontend
  artifacts/                # trained model + encoders + catalog embeddings (gitignored)
  deck/                     # final presentation (PDF)
```

---

## 6. Work split (5 members)

Everyone must speak in the presentation, so each owns a slide segment too.

1. **Data & preprocessing** — MovieLens EDA, encoding, sequence builder, leave-one-out
   split, min-count filter. Deck: dataset + problem framing.
2. **Model** — dual-head architecture, training, tuning, export. Deck: architecture justification.
3. **Evaluation & baselines** — ranking metrics, popularity/recency baselines,
   overfitting analysis. Deck: results + why the RNN wins.
4. **Frontend** — React dashboard, UX, feedback reactions, taste-drift viz,
   serendipity dial. Deck: live demo.
5. **Integration + personal data + deck owner** — Letterboxd/Netflix parsing + title
   resolution, TV catalog, wire model→app, README, business case. Deck: business value + close.

---

## 7. Phased plan (~2 weeks)

**Week 1 — working model end to end**

- **Phase 0 — Setup (day 1):** repo, GitHub, requirements, download/place MovieLens,
  agree on the `recommend(history) -> top_n` interface so frontend and model build in parallel.
- **Phase 1 — Data (days 1-3):** MovieLens EDA, preprocessing, sequence builder,
  leave-one-out split, min-count filter. Output: arrays + encoders + genre matrix.
- **Phase 2 — Model v1 (days 3-5):** dual-head GRU end-to-end (even if weak) to lock
  the pipeline and export artifacts.
- **Phase 3 — Eval + baselines (days 4-6):** ranking metrics + popularity/recency baselines.

**Week 2 — make it real and present it**

- **Phase 4 — Tune (days 6-8):** sequence length, embedding sizes, layers, dropout;
  beat the baselines. Keep tuning shallow.
- **Phase 5 — Personal data + catalog (days 6-9, parallel):** parse Letterboxd +
  Netflix, resolve titles, ingest `imdb_tvshows.csv`, build content embeddings.
- **Phase 6 — Frontend + creative (days 7-12, parallel):** React app + FastAPI, feedback
  re-rank, taste-drift viz, serendipity dial, "why this", real-time integration hardening.
- **Phase 7 — Deck + rehearsal (days 11-14):** build deck, time the 15-min run, record
  a backup demo, rehearse demo + Q&A, freeze the repo.

---

## 8. Risks & mitigations

- **Popularity bias:** genre+rating hybrid, min-count filter, top-N diversity, and
  reported baselines so the uplift is visible.
- **New metric family (ranking):** budget time in Phase 3; one owner.
- **TV title resolution (Spanish + partial catalog coverage):** scope to a curated
  ~20-30 show subset for the demo; accept partial coverage; state as a limitation.
- **No personal chronology (Letterboxd):** use it as a taste profile; construct a
  plausible recent-history input or let the user order a few titles in the app.
- **Live demo failure (10%):** pre-load artifacts, recorded backup, rehearse on the
  demo machine, freeze the repo before presenting.
- **Cold start (empty history):** fall back to popular / genre-seeded recs in the app.

---

## 9. Tech stack

- Python, TensorFlow/Keras (matches the course notebooks), pandas/numpy, scikit-learn
  (metrics/baselines/nearest-neighbor), React + Vite + Tailwind (frontend), FastAPI (serving), GitHub (repo + README).
