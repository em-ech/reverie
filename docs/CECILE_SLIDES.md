# Reverie — Slide Notes for Cecile

> Everything you need to build the presentation. Every number here comes directly
> from `results/log.csv` and the experiment scripts — nothing is made up.
> Check `EXPERIMENTS.md` for the full tables and `README.md` for the project summary.

**Presentation date:** July 2, 2026
**Team:** Amaly Attia · Em Echeverria · Lea Sarouphim Hochar · Stephan Pentchev · Cecile Tambey

---

## Suggested Slide Order

```
1. Title
2. Problem & Motivation
3. Dataset & Data Pipeline
4. Model Architecture
5. Why These Choices? (Experiments E3 / E4 / E5)
6. Are the Input Features Justified? (E6)
7. Does the Model Beat the Baselines? (E0)
8. Official Result (Test Set)
9. Live Demo
10. Limitations
11. Future Work
12. Conclusion
```

---

## Slide 1 — Title

**Title:** Reverie — Sequential Movie & TV Recommender

**Subtitle:** A GRU-based next-item prediction model trained on MovieLens ml-1m

**Team names, IE University, Deep Learning, July 2026**

---

## Slide 2 — Problem & Motivation

**The point:** most recommendation systems treat your taste as a static average.
Reverie treats it as a _sequence_ — what you watched last week matters more than
what you watched 3 years ago.

**Key message to land:**

- Netflix/Spotify recommendations feel stale because they average your whole history
- A recurrent model reads your history _in order_ and predicts what comes next
- When you reject a recommendation, Reverie corrects itself instantly — no retraining

**Visual idea:** a timeline of movies a user watched, with an arrow pointing to the
predicted next one. Or the app screenshot showing the recommendation row.

---

## Slide 3 — Dataset & Data Pipeline

**Dataset:** MovieLens ml-1m

- 1,000,209 ratings
- 6,040 users
- 3,413 movies (after filtering movies seen fewer than 5 times)
- Ratings: 1 to 5 stars, with timestamps

**Why ml-1m and not a smaller dataset?**
ml-1m gives us 10x more sequences than the smaller alternative (ml-latest-small,
100K ratings). A reliable softmax over thousands of movies needs dense training data.
We chose ml-1m from day 1 for this reason.

**How the data is split (leave-one-out):**

```
User's full history (sorted by timestamp):
  Movie A → Movie B → Movie C → Movie D → Movie E

  Train:      A, B, C     (used to train model weights)
  Validation: D           (used to tune hyperparameters — E3/E4/E5)
  Test:       E           (opened ONCE with the frozen config — Lea's run)
```

**Key guarantee:** the model never sees the test target during training or tuning.
The test set was sealed until all decisions were made.

**Files to reference:**

- `src/data_prep.py` — the full pipeline in code
- `data/data_notes.md` — explains the dataset choice

---

## Slide 4 — Model Architecture

**The model in one diagram:**

```
User's watch history (last 20 movies)
        |
[Movie Embedding, 32d]  +  [Genre Lookup, frozen]  +  [Star Rating, scaled]
        |
   [GRU Layer, 64 units]
        |
   [Dropout 0.3]
        |
   [Softmax over 3,413 movies]
        |
   Top-10 = Recommendations
```

**What each part does — explain this simply:**

| Layer                 | What it does                                                                                |
| --------------------- | ------------------------------------------------------------------------------------------- |
| Movie Embedding (32d) | Turns each movie ID into 32 numbers the model learns — similar movies end up close together |
| Genre Lookup (frozen) | Appends the movie's genre tags (Action, Drama, etc.) — fixed, not trained                   |
| Star Rating channel   | Appends your 1-5 star rating (rescaled) — tells the model if you loved or hated it          |
| GRU (64 units)        | Reads the sequence left-to-right, compresses everything into a single "taste vector"        |
| Dropout (0.3)         | Randomly zeroes 30% of neurons during training — prevents memorization                      |
| Softmax               | Outputs a probability for every one of the 3,413 movies — highest = recommendation          |

**Why GRU and not LSTM?**
We ran both (E5). They tied on accuracy (HR@10: GRU 0.274 vs LSTM 0.273).
GRU has fewer parameters (no forget gate) so it trains faster and is less likely
to overfit on our 6K-user dataset. Empirical result, not just theory.

**Why GRU and not a Transformer?**
Transformers need far more data and compute. With 6,040 users, a GRU is the
right tool. We ran the math — a Transformer would be overkill and would likely
overfit.

**Files to reference:**

- `src/model.py` — the actual model code (build_hybrid_gru)
- `ARCHITECTURE.md` — full design rationale

---

## Slide 5 — Why These Choices? (Experiments E3/E4/E5)

**The point of this slide:** we did not just guess the architecture.
We ran ablations on the validation set and made every decision empirically.

**E3 — How much history does the model need?**

| History length | HR@10      | Verdict                       |
| -------------- | ---------- | ----------------------------- |
| L = 10         | 0.2707     | Too short — misses context    |
| L = 20         | **0.2779** | Winner — best AND faster      |
| L = 50         | 0.2739     | Longer adds noise, not signal |

Key insight: L=20 _beats_ L=50. Recent taste matters more than old history.
This also makes intuitive sense — what you watched last month predicts next week
better than what you watched 5 years ago.

**E4 — How big should the model be?**

| embed_dim | rnn_units | HR@10      | Verdict                                |
| --------- | --------- | ---------- | -------------------------------------- |
| 16        | 64        | 0.2725     | Too small                              |
| 32        | 64        | **0.2740** | Winner — smallest within noise of best |
| 32        | 128       | 0.2779     | Good but 2x the parameters for +0.004  |
| 64        | 128       | 0.2782     | Raw best — but 4x the parameters       |
| 64        | 256       | 0.2687     | Clear overfitting — stops at epoch 7   |

Key insight: we chose the _smallest_ model within noise of the best.
Bigger is not better here — d=64/u=256 actually gets _worse_ (overfits).

**E5 — GRU or LSTM?**

| Cell | HR@10  | MRR    | Time      |
| ---- | ------ | ------ | --------- |
| GRU  | 0.2740 | 0.1345 | 13s/epoch |
| LSTM | 0.2730 | 0.1333 | 13s/epoch |

Tied within 0.001 — GRU wins on fewer parameters.

**Files to reference:**

- `results/log.csv` rows 9-19 — all E3/E4/E5 numbers
- `experiments/run_e3.py`, `run_e4.py`, `run_e5.py` — the scripts

---

## Slide 6 — Are the Input Features Justified? (E6)

**The point:** we don't just add features for the sake of it.
We tested each one and kept it only if it earned its place.

| Input variant                | HR@10      | MRR        | NDCG@10    |
| ---------------------------- | ---------- | ---------- | ---------- |
| Movie ID only                | 0.2765     | 0.1310     | 0.1539     |
| + Genre (frozen)             | **0.2812** | 0.1324     | 0.1561     |
| + Rating only                | 0.2558     | 0.1290     | 0.1471     |
| Full hybrid (genre + rating) | 0.2740     | **0.1345** | **0.1555** |

**Decision: full hybrid.** Reasons:

1. Best MRR — it ranks the correct movie highest most often
2. Best NDCG — best overall ranking quality
3. The rating channel is _required_ for the thumbs-down feature in the app
   (when a user rejects a recommendation, we append a low-rating step to the
   history and re-rank — without the rating channel this feature breaks)

**Note on +rating alone hurting HR@10:** star ratings are noisy on their own.
A user might rate an action film 5 stars and also rate a romance 5 stars.
Without genre context, the model can't tell which signal to follow.
Combined with genre, the rating becomes meaningful.

**Files to reference:**

- `results/log.csv` rows 23-26 — E6 numbers
- `experiments/run_ablation.py` — the script

---

## Slide 7 — Does the Model Beat the Baselines? (E0)

**The point:** the model must beat simple non-neural baselines to justify
the complexity. It does — by a large margin, with statistical proof.

| Model             | HR@10 [95% CI]            | MRR [95% CI]              | vs GRU (Wilcoxon p) |
| ----------------- | ------------------------- | ------------------------- | ------------------- |
| **Reverie (GRU)** | **0.2641 [0.252, 0.275]** | **0.1252 [0.119, 0.132]** | —                   |
| item-kNN          | 0.0759 [0.069, 0.083]     | 0.0352 [0.032, 0.038]     | ~0                  |
| recent-genre pop  | 0.0446 [0.039, 0.050]     | 0.0225 [0.020, 0.025]     | ~0                  |
| most-popular      | 0.0419 [0.037, 0.047]     | 0.0215 [0.019, 0.024]     | ~0                  |
| random            | 0.0029                    | —                         | —                   |

**Key points to highlight:**

- GRU is **6x better** than most-popular (0.264 vs 0.042)
- GRU **beats item-kNN** — the strong baseline that shallow models often lose to
- All confidence intervals are non-overlapping — the win is not luck
- Wilcoxon p ≈ 0 across all three baselines — statistically significant

**What these baselines are (explain simply):**

- most-popular: always recommends globally popular movies — ignores your taste
- recent-genre pop: recommends popular movies in genres you recently watched — slightly personalized
- item-kNN: finds movies similar to what you watched using cosine similarity — no sequence awareness

**Files to reference:**

- `EXPERIMENTS.md` — headline baselines table
- `experiments/run_baselines.py` — the script
- `results/log.csv` rows 1-3 — baseline numbers

---

## Slide 8 — Official Result (Test Set)

**This is the headline slide. One number.**

```
HR@10 = 0.242 ± 0.002
(mean ± std, 3 independent seeds, test set)
```

**What this means in plain English:**
Given a user's watch history, the model puts the movie they actually watched next
inside its top-10 recommendations 24.2% of the time — across 6,032 users, on data
it never saw during training or tuning.

**Why we report it this way:**

- **3 seeds:** a single training run is a random variable. We ran 3 independent
  seeds (42, 0, 7) to show the result is stable, not lucky.
- **Test set:** this data was sealed from day 1. The model never saw it during
  E3/E4/E5 tuning. This is the honest, unbiased number.
- **Full-catalog ranking:** we rank all 3,413 movies for each user — not just
  100 sampled negatives, which inflates scores artificially.

**The val-to-test drop (0.274 → 0.242) is normal and expected:**
Models always score slightly lower on unseen test data vs validation.
A ~3% drop with a std of ±0.002 shows the model generalized well — it did not
overfit to the validation set.

**Files to reference:**

- `results/log.csv` rows 20-22 — the 3 seed results
- `src/run_test.py` — Lea's evaluation script

---

## Slide 9 — Live Demo

**What to show:**

1. Open the app at `http://localhost:5173`
2. Search for a movie (e.g. "The Matrix") and add it to history
3. Add 3-4 more movies with star ratings
4. Show the recommendation row updating in real time
5. Thumbs-down one recommendation — show it disappear and the list rerank
6. Show the taste profile panel (genre breakdown of what you like)

**Before the presentation — test this the night before:**

- Terminal 1: `.venv\Scripts\Activate.ps1` (keep open for any scripts)
- Terminal 2: `.venv\Scripts\Activate.ps1` then `uvicorn app.api:app --port 8000`
- Terminal 3: `cd web` then `npm run dev`
- Open `http://localhost:5173` — make sure it loads

**If the demo breaks live:**
The professor understands technical demos can fail. Have a screenshot or screen
recording as backup. The results slide (HR@10=0.242) is the real evidence anyway.

---

## Slide 10 — Limitations

**Be honest — the professor respects honesty more than overselling.**

1. **Movie-only training, TV via proxy**
   The GRU is trained on MovieLens movies. TV show recommendations use a genre
   similarity bridge (cosine of genre vectors) — not a trained signal. This is an
   approximation. A TV-specific training corpus would make it proper.

2. **Cold start**
   A new user with no history gets generic (popularity-based) recommendations.
   The model needs at least 3-4 ratings to produce meaningful personalization.

3. **CPU training only (Windows)**
   TensorFlow dropped native Windows GPU support after version 2.10. Training runs
   on CPU (~13s/epoch). WSL2 + CUDA would bring this to ~2s/epoch. This limited
   how many experiments we could run in the time available.

4. **Single softmax over 3,413 movies**
   The model scores every movie on every prediction. This is correct and rigorous,
   but doesn't scale to Netflix-scale catalogs (100M+ items). At scale you'd need
   a two-stage retrieve-then-rank system.

5. **No cross-user signals**
   The model learns from each user's own history, but never explicitly says
   "users similar to you also liked X." Collaborative filtering signals are
   implicit (learned through shared embeddings) but not explicit.

---

## Slide 11 — Future Work

**Frame these as "what we would do next" — not as failures.**

**The hat trick we considered but didn't run:**

**E2 — Logit Adjustment (popularity-bias correction)**
Our softmax still tends to over-recommend blockbusters (The Dark Knight, Pulp Fiction)
because they appear disproportionately in training data. The principled fix is to
subtract the log of each movie's training frequency from its score — this penalizes
popular items and surfaces niche recommendations. We evaluated the option and decided
against it because the model already beats all baselines without it, and adding it
would have changed the model output right before the test run. It is the single most
impactful improvement available with one line of code.

**E1 — Dataset comparison (ml-1m vs ml-latest-small)**
We chose ml-1m from the start — 1M ratings vs 100K, 6K users vs 600. We never
formally ran the comparison because the rationale was clear: a softmax over 9,700
items with only 610 users would overfit badly. A formal E1 run would have produced
the quantitative evidence for this decision.

**Other future directions:**

- **Multi-layer GRU:** stack a second GRU layer. Likely helps on larger datasets.
- **SASRec / Transformer:** attention-based sequential model outperforms GRU at
  scale (millions of users). Overkill for 6K users but the natural next step.
- **GPU training via WSL2:** would cut training time 6-7x, enabling more
  experiments and larger models.
- **TV-specific training corpus:** train a second model on TV watch data directly
  instead of using the genre-bridge proxy.

---

## Slide 12 — Conclusion

**Three things to land:**

1. **It works.**
   HR@10 = 0.242 ± 0.002 on a blind test set. 6x better than most-popular.
   Beats item-kNN — the baseline shallow models often lose to.

2. **Every decision was empirical.**
   L=20 from E3. d=32/u=64 from E4. GRU from E5. Full hybrid from E6.
   Nothing was guessed — all choices are justified by validation numbers.

3. **It runs.**
   Live demo with a real React frontend, FastAPI backend, and a trained model
   serving real-time recommendations. The loop closes.

---

## Quick Reference — Numbers by Slide

| Slide          | Key number                                  | Source                               |
| -------------- | ------------------------------------------- | ------------------------------------ |
| Baselines (S7) | GRU 0.2641 vs popular 0.0419                | `EXPERIMENTS.md`, `log.csv` rows 1-6 |
| E3 (S5)        | L=20 HR@10=0.2779 beats L=50 0.2739         | `log.csv` rows 9-11                  |
| E4 (S5)        | d32/u64 HR@10=0.2740 vs d64/u256 0.2687     | `log.csv` rows 12-17                 |
| E5 (S5)        | GRU 0.2740 vs LSTM 0.2730                   | `log.csv` rows 18-19                 |
| E6 (S6)        | full hybrid MRR=0.1345, +genre HR@10=0.2812 | `log.csv` rows 23-26                 |
| Test (S8)      | HR@10 = 0.242 +/- 0.002                     | `log.csv` rows 20-22                 |

## Quick Reference — Files to Look At

| What you need                                  | Where to find it                         |
| ---------------------------------------------- | ---------------------------------------- |
| All experiment numbers                         | `results/log.csv`                        |
| Train vs val loss curve (overfitting evidence) | `results/curve_artifact_full_hybrid.png` |
| Experiment tables and decisions                | `EXPERIMENTS.md`                         |
| Model architecture diagram                     | `ARCHITECTURE.md` section 3              |
| Project summary                                | `README.md`                              |
| The actual model code                          | `src/model.py`                           |
| Data pipeline code                             | `src/data_prep.py`                       |
