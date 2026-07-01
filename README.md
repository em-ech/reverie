# Reverie — Movie & TV Recommender

Reverie reads a viewer's watch history and predicts what they will enjoy watching next,
across **both movies and TV shows**. It pairs two neural models with a cinematic web app:

1. **A sequential GRU** (the course deliverable) that learns evolving taste from watch
   _sequences_ and re-ranks the moment you reject a pick.
2. **A neural collaborative-filtering model** (post-submission product work) trained on
   ~9.9M Letterboxd ratings, which fixes what the GRU could not do: recommend modern
   films it was never trained on, and slot a brand-new person in as a user.

Built for the IE University Deep Learning final project, then extended into a real product.

**Team:** Amaly Attia · Em Echeverria · Lea Sarouphim Hochar · Stephan Pentchev · Cecile Tambey

---

## The business case

**Problem.** Streaming choice overload. Households now juggle four or more services and
tens of thousands of titles, and the average viewer spends roughly **10 minutes per
session just deciding** what to watch (Nielsen), with many giving up without watching
anything. For couples and groups the negotiation is worse. That wasted time is the user's
pain; for platforms it surfaces as disengagement and churn.

**Value proposition.**

- **Saves time.** Reverie replaces ten minutes of scrolling with a ranked shortlist in
  under a second. Its high-confidence "you will love it" picks held a **precision of 1.00**
  on held-out real viewers, trustworthy enough to act on immediately.
- **Solves the group decision.** The Blend intersects two people's recommendations so
  couples and friends skip the back and forth.
- **Drives retention (platform framing).** Personalization is the churn lever. Netflix has
  publicly attributed roughly **$1B per year** to its recommender through reduced churn; a
  single point of churn reduction on 1M subscribers at €15 per month is about **€1.8M per
  year** retained. Reverie is that engine, productized. (Industry figures, illustrative of
  the lever, not a measured claim about this model.)

**Who it is for.** Consumers drowning in streaming choice (the MVP here), and, as a B2B
engine, any platform that lives or dies on retention.

---

## What it does

- **Sequential model (the GRU):** a recurrent network trained on MovieLens watch
  sequences. It models taste as a _sequence_ — what you watched recently predicts what
  you watch next far better than a static average profile.
- **Collaborative model (NCF):** user and movie **embeddings** plus bias terms and a
  frozen content tower, trained on millions of Letterboxd ratings. It beats every naive
  baseline globally and for 94% of individual users, and its high-confidence "you will
  love it" picks held a precision of 1.00 across two real held-out viewers. See
  [`notebooks/ncf_collaborative.ipynb`](notebooks/ncf_collaborative.ipynb).
- **Movies + TV:** the GRU softmax head ranks movies directly; a derived, mean-centered
  **taste vector** scores any title (movie or TV) by genre similarity, bridging to a
  catalog it was never trained on.
- **Learns from mistakes:** a thumbs-down is appended to the live history as a low
  rating; the model re-runs and the recommendations re-rank instantly — no retraining.
- **A real product app:** accounts, a swipe-or-cards watch-history builder (save to a
  watchlist, rate what you have seen, the deck tunes to each pick), a template taste
  blurb, friends, and a "Blend" that intersects two people's recommendations.

See [`PROJECT_PLAN.md`](PROJECT_PLAN.md), [`ARCHITECTURE.md`](ARCHITECTURE.md),
[`AUDIT.md`](AUDIT.md), and [`EXPERIMENTS.md`](EXPERIMENTS.md) for the full design,
adversarial pre-build review, and experiment protocol.

## Results (MovieLens ml-1m, validation)

The RNN beats every baseline with non-overlapping 95% confidence intervals and paired
Wilcoxon p ≈ 0 — including item-kNN, the strong baseline shallow RNNs often lose to.

| Model             | HR@10     | MRR       |
| ----------------- | --------- | --------- |
| **Reverie (GRU)** | **0.257** | **0.120** |
| item-kNN          | 0.076     | 0.035     |
| recent-genre pop  | 0.045     | 0.023     |
| most-popular      | 0.042     | 0.022     |

Feature ablation (E6): genre lifts HR@10 to **0.281**; rating alone hurts HR@10 but the
full hybrid (genre + rating) wins on MRR (**0.1345**) and is the chosen configuration —
it also powers the app's thumbs-down feedback feature. Official test result (3 seeds):
**HR@10 = 0.242 ± 0.002**.

## Results (collaborative model, Letterboxd)

The collaborative NCF model is a rating regression (1 to 10, loss MSE, reported as RMSE)
trained on ~9.9M ratings. It beats all three naive baselines on the held-out test set,
and beats the toughest per-movie baseline for the large majority of individual users.

| Predictor            | Test RMSE |
| -------------------- | --------- |
| global-mean baseline | 2.072     |
| user-mean baseline   | 1.963     |
| movie-mean baseline  | 1.619     |
| **neural net (NCF)** | **1.372** |

Per user, the net beats the movie-mean baseline for **94%** of viewers (median per-user
RMSE 1.320). Two real people were slotted in and their most recent films held out by
date: exact rating prediction for a brand-new, lightly-rated user sits near their personal
average (the cold-start floor), but the model's high-confidence **"will they love it
(4+ stars)"** picks had **precision 1.00** for both, zero false alarms — the trustworthy
short list a recommender actually needs. Full writeup:
[`notebooks/ncf_collaborative.ipynb`](notebooks/ncf_collaborative.ipynb).

## Architecture

```
watch history ─▶ [movie emb ⊕ genre lookup ⊕ rating] ─▶ GRU ─▶ softmax (next movie)
                                                          │
                                                          └─▶ taste vector  (content match → TV + movies)
```

Built with the audit's correctness contract: all-prefix training windows with
leave-one-out evaluation, train-only statistics (no leakage), reserved padding index,
explicit masking, and full-catalog ranking metrics with confidence intervals.

## Tech stack

- **Model / API:** Python, TensorFlow/Keras, FastAPI, scikit-learn, pandas/numpy
- **Frontend:** React + Vite + TypeScript + Tailwind + shadcn/ui + recharts

## Project structure

```
src/                core library: data prep, model, evaluate, recommend, baselines, track
  train.py          GRU training pipeline -> writes artifacts/
  run_test.py       official GRU test evaluation (frozen config, 3 seeds)
  ncf_model.py      collaborative model (user + movie embeddings + content tower)
  ncf_data.py       loaders for the NCF artifacts
  ncf_baselines.py  global / movie / user mean baselines + RMSE
  ncf_recommend.py  serving seam: rank the modern catalog by your nearest favorites
scripts/            pipelines: build_ncf_dataset, train_ncf, eval_ncf, build_modern_catalog
app/                FastAPI service, layered:
  api.py            app factory + router wiring
  routers/          auth, catalog (+ browse), history, watchlist, friends, blend
  services/         recommend, friend, blend, taste, blurb, copy (business logic)
  models.py db.py   SQLAlchemy ORM (User, HistoryItem, WatchlistItem, Friendship) + session
  enrich.py         movieId -> title/genre/poster/rating (single authority; modern or ml-1m)
web/                React frontend (Vite + Tailwind + shadcn/ui)
notebooks/          temporal-validation + ncf_collaborative writeups
experiments/        one-off GRU research scripts (E0, E3, E4, E5, baselines, ablation) — done
presentation/       the final slide deck (reverie-deck.html, self-contained) + team PDF/PPTX
artifacts/          trained weights + encoders + NCF artifacts (generated locally, not committed)
data/               datasets (not committed — see below)
*.md                plan, architecture, audit, experiment protocol
```

## Presentation

The final deck is a self-contained, animated HTML file: open
[`presentation/reverie-deck.html`](presentation/reverie-deck.html) in a browser, press
`f` for fullscreen, and use the arrow keys. It explains both models with the course
concepts, a full system architecture diagram, and the real result graphs. Print to PDF
(`Cmd+P`) for a static copy, or screen-record / export the animations to GIF for tools
like Canva.

## Data (not included)

No datasets are committed. The training data is public; personal demo data is private
and intentionally excluded.

- **Training (primary):** [MovieLens `ml-1m`](https://grouplens.org/datasets/movielens/1m/)
  — 1M ratings, 6,040 users.

  ```bash
  curl -O https://files.grouplens.org/datasets/movielens/ml-1m.zip
  unzip ml-1m.zip -d data/
  ```

- **TV catalog:** an IMDb TV-shows CSV (title, genres, description) in `data/`.
- **Personal demo (optional):** a user's own Letterboxd / Netflix export. These contain
  private information and are **never** committed (see `.gitignore`).

### Modern catalog (the app's default)

The app serves a **modern** catalog (post-2000 films) from the collaborative model by
default (`REVERIE_CATALOG_MODE=modern`). Build it once from the Letterboxd metadata and
the trained NCF artifacts:

```bash
python -m scripts.build_modern_catalog --movies ~/Downloads/movie_data.csv \
  --ncf artifacts/ncf --limit 8000 --min-votes 50 --out artifacts/modern
```

Posters come straight from the dataset (Letterboxd CDN), so no TMDB key is needed. Set
`REVERIE_CATALOG_MODE=ml1m` to fall back to the original MovieLens GRU demo. Switching
modes changes the movie id space (tmdb ids vs MovieLens ids), so start a fresh `reverie.db`
when you switch.

## Setup & run

### First time only

```bash
# Backend (Python 3.11; matches the course conda env)
pip install -r requirements.txt

# 1. Train the model -> writes artifacts/
python -m src.train

# 2. Start the API (Terminal 2)
uvicorn app.api:app --port 8000
```

**Terminal 3 - Frontend (PowerShell):**

```powershell
# 3. Frontend
cd web
npm install
npm run dev
```

Then open <http://localhost:5173>

---

### Every session after (demo / repeat runs)

No need to retrain or reinstall. Just start the API and frontend in two terminals:

**PowerShell:**

```powershell
# Terminal 1 - API (activate venv first, then run)
.venv\Scripts\Activate.ps1
uvicorn app.api:app --port 8000
```

```powershell
# Terminal 2 - Frontend
cd web
npm run dev
```

**Bash (Git Bash / Mac / Linux):**

```bash
# Terminal 1 - API (activate venv first, then run)
source .venv/Scripts/activate
uvicorn app.api:app --port 8000
```

```bash
# Terminal 2 - Frontend
cd web && npm run dev
```

Then open <http://localhost:5173>

## Future considerations

Things explored or considered during the project but not implemented — good starting
points if the project is extended beyond the course:

- **ml-latest-small comparison (E1):** we chose ml-1m from the start based on size
  (1M ratings vs 100K) and never ran a formal comparison. ml-latest-small is too sparse
  for a reliable softmax over 9,700 items — the decision was clear enough not to need an experiment.
- **Logit adjustment (popularity-bias correction):** subtract the log train-prior from
  the softmax logits to stop the model over-recommending blockbusters. The current model
  already beats all baselines without it, but this would be the next thing to try if
  diversity of recommendations matters more than raw HR@10.
- **Multi-layer GRU:** stack a second GRU layer on top of the first. Not done because a
  single layer was sufficient on this dataset size (6K users) and adding layers risks
  overfitting.
- **GPU training via WSL2:** the current setup trains on CPU (TensorFlow dropped native
  Windows GPU support after v2.10). WSL2 with CUDA would cut training time from ~13s/epoch
  to ~2s/epoch on the same machine.
- **Transformer-based recommender (e.g. SASRec):** attention-based sequential model that
  outperforms GRU on very large datasets. Overkill for 1M ratings but worth exploring at
  scale.

---

## Privacy

Personal streaming exports include billing, payment, IP, account, and device records.
Reverie only ever reads viewing/ratings files, and `.gitignore` blocks all personal and
sensitive data from this public repository.

---

## Data sources and acknowledgments

Reverie is built on public data, with thanks to:

- **MovieLens (GroupLens, University of Minnesota)** — the `ml-1m` ratings used to train the
  sequential GRU model.
- **The Movie Database (TMDB)** — posters, backdrops, genres, ratings, and streaming
  availability via the TMDB API. This product uses the TMDB API but is not endorsed or
  certified by TMDB.
- **samlearner, "Letterboxd Movie Ratings Data" (Kaggle)** — the multi-user Letterboxd
  ratings used for the collaborative recommender.
- **gsimonx37, "Letterboxd" (Kaggle)** — Letterboxd movie metadata (cast, crew, themes,
  posters) used for the modern catalog and content features.

Personal watch histories (Letterboxd / Netflix exports) are never committed and are used
only locally.
