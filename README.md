# Reverie — Sequential Movie & TV Recommender

Reverie reads a viewer's watch history, learns their evolving taste with a recurrent
neural network, and predicts what they will enjoy watching next — across **both movies
and TV shows**. When the user rejects a recommendation, the model corrects itself in
real time.

Built for the IE University Deep Learning final project.

**Team:** Amaly Attia · Em Echeverria · Lea Sarouphim Hochar · Stephan Pentchev · Cecile Tambey

---

## What it does

- **Backend (the AI model):** a GRU recurrent network trained on the MovieLens watch
  sequences. It models taste as a _sequence_ — what you watched recently predicts what
  you watch next far better than a static average profile.
- **Movies + TV:** the softmax head ranks movies directly; a derived, mean-centered
  **taste vector** scores any title (movie or TV) by genre similarity, so the system
  bridges to a TV catalog it was never trained on.
- **Learns from mistakes:** a thumbs-down is appended to the live history as a low
  rating; the model re-runs and the recommendations re-rank instantly — no retraining.
- **Frontend:** a cinematic React dashboard (dark, crimson, Netflix-style rows) where a
  non-technical user builds a watch history and gets real-time recommendations plus a
  visual taste profile.

See [`PROJECT_PLAN.md`](PROJECT_PLAN.md), [`ARCHITECTURE.md`](ARCHITECTURE.md),
[`AUDIT.md`](AUDIT.md), and [`EXPERIMENTS.md`](EXPERIMENTS.md) for the full design,
adversarial pre-build review, and experiment protocol.

## Results so far (MovieLens ml-1m, validation)

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
src/             core library: data prep, model, evaluate, recommend, baselines, track
  train.py       production training pipeline -> writes artifacts/
  run_test.py    official test evaluation (frozen config, 3 seeds) -> results slide number
app/api.py       FastAPI service (/health, /catalog, /recommend)
web/             React frontend (Vite + Tailwind + shadcn/ui)
experiments/     one-off research scripts (E0, E3, E4, E5, baselines, ablation) — done
artifacts/       trained model weights + encoders (generated locally, not committed)
data/            datasets (not committed — see below)
*.md             plan, architecture, audit, experiment protocol
```

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

## Setup & run

```bash
# Backend (Python 3.11; matches the course conda env)
pip install -r requirements.txt

# 1. Train the model -> writes artifacts/
python -m src.train

# 2. Start the API
uvicorn app.api:app --port 8000

# 3. Start the frontend (separate terminal)
cd web && npm install && npm run dev   # http://localhost:5173
```

For a single-process live demo, build the frontend (`npm run build`) and have FastAPI
serve the static bundle.

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
