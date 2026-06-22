# Reverie — Sequential Movie & TV Recommender

A deep learning MVP that reads a viewer's watch history, learns their evolving taste,
and recommends what to watch next across **both movies and TV shows**. When the user
rejects a recommendation, the model corrects itself in real time.

Built for the IE University Deep Learning final project.

## What it does

- **Backend:** a recurrent neural network (GRU/LSTM) trained on the MovieLens watch
  sequences. It models taste as a _sequence_ — recent watches predict the next watch.
- **TV + movies:** a dual-head design lets the model recommend across both a movie
  catalog and a TV-show catalog by matching on content (genre) rather than exact titles.
- **Learns from mistakes:** thumbs-up / thumbs-down feedback is fed back into the live
  sequence and the recommendations re-rank instantly.
- **Frontend:** an interactive Streamlit dashboard. A non-technical user builds or
  imports a watch history and gets real-time recommendations.

See [`PROJECT_PLAN.md`](PROJECT_PLAN.md) for the full design, data decisions, and rubric mapping.

## Architecture

```
watch history ─▶ [movie emb ⊕ genre ⊕ rating] ─▶ GRU/LSTM ─▶ hidden state
                                                       ├─▶ next-movie softmax  (quantitative eval)
                                                       └─▶ taste vector        (content match → TV + movies)
```

## Project structure

```
src/        model, training, evaluation, data prep, inference
app/        Streamlit dashboard
notebooks/  EDA, preprocessing, training, evaluation
data/        datasets (NOT included — see below)
artifacts/  trained model + encoders (generated locally, not committed)
deck/        final presentation
```

## Data (not included)

No datasets are committed to this repo. The training data is public; the personal
demo data is private and intentionally excluded.

- **Training (primary):** [MovieLens `ml-1m`](https://grouplens.org/datasets/movielens/1m/)
  — download and unzip into `data/ml-1m/` (~6 MB, 1M ratings, 6,040 users). This is the
  primary corpus; `ml-latest-small` is a fallback only (see `ARCHITECTURE.md` / `AUDIT.md`).

  ```bash
  curl -O https://files.grouplens.org/datasets/movielens/ml-1m.zip
  unzip ml-1m.zip -d data/
  ```

- **TV catalog:** an IMDb TV-shows CSV (title, genres, description) in `data/`.
- **Personal demo (optional):** a user's own Letterboxd and/or Netflix export. These
  contain private information and are **never** committed (see `.gitignore`).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# 1. Train the model (writes artifacts/)
python -m src.train

# 2. Start the API (FastAPI, serves recommend())
uvicorn app.api:app --port 8000

# 3. Start the React frontend (separate terminal)
cd web && npm install && npm run dev   # http://localhost:5173
```

The frontend (`web/`, React + Vite + Tailwind + shadcn/ui) calls the API's
`/recommend` and `/catalog` endpoints. For a single-process demo, build the
frontend (`npm run build`) and have FastAPI serve the static bundle.

## Privacy

Personal streaming exports include billing, payment, IP, account, and device records.
This project only ever reads viewing/ratings files, and `.gitignore` blocks all
personal and sensitive data from being committed to this public repository.

## Team

Five-member group project. See the presentation deck for individual contributions.
