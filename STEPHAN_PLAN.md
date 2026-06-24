# Stephan's Plan — The AI Model

> **Your role:** Lock the final model config so Lea can run the official test and Cecile can save the final artifact.
> They are blocked until you finish Step 4. Do Steps 0–4 first, then Step 5.

---

## The Big Picture of What You Own

You are responsible for answering three questions empirically (on validation data):
1. How much history does the model need to read? → **E3**
2. How big should the model be? → **E4**
3. GRU or LSTM? → **E5**

Once you have answers, you write them down (locked config), and everyone else can move.

---

## Step 0 — Verify the full pipeline works

**Command:**
```bash
python -m src.train
```

**What it does:**
Loads `data/ml-1m/ratings.dat`, builds the dataset (1M ratings, 6,040 users, 3,706 movies),
trains the full hybrid GRU with early stopping, then exports four files to `artifacts/`:
- `weights.weights.h5` — the trained model weights
- `model_config.json` — architecture settings (n_items, max_len, embed_dim, etc.)
- `movie_index.json` — maps internal IDs back to real MovieLens movie IDs
- `genre_matrix.npy`, `genre_marginal.npy`, `popularity_train.npy` — supporting data

**What it contributes:**
- Proves the entire data → model → artifacts pipeline runs on your machine
- Creates the `artifacts/` folder so Em's frontend and the FastAPI backend can actually serve recommendations
- **Em and Cecile are unblocked the moment these files exist**
- Result is logged automatically to `results/log.csv`

**Expected output:**
```
items=3706  train_windows=~160,000
val: HR@10=0.2735  MRR=0.1348  NDCG@10=0.1556
Artifacts written to artifacts/
```

- [x] Step 0 complete — `artifacts/` folder has all 6 files

---

## Step 1 — E3: How much history does the model need?

**Script to write:** `src/run_e3.py`  
**Command (once written):**
```bash
python -m src.run_e3
```

**What it does:**
Trains three versions of the model with different history window sizes, everything else fixed:
- `max_len = 10` — model only sees the last 10 things you watched
- `max_len = 20` — model sees the last 20
- `max_len = 50` — model sees the last 50 (current default)

Evaluates each on validation HR@10, MRR, NDCG@10 and prints a comparison table.

**What it contributes:**
- Answers: "does watching more history actually help, or does the last 10 items have all the signal?"
- Lets you pick the **smallest window within noise of the best** — simpler and faster
- The answer becomes a slide: "we tested L=10/20/50 and found L=X was the sweet spot"
- This is the architecture justification the professor looks for

**Decision rule:** if L=20 is within ~0.5% HR@10 of L=50, go with L=20 (simpler model).

- [x] `src/run_e3.py` written
- [x] E3 run completed
- [x] Best `max_len` value noted: `max_len = 20`

E3 Results — What they mean

max_len	HR@10	Verdict

10	0.2707	Too short — missing context

20	0.2779	Winner — best AND faster

50	0.2739	Longer doesn't help

The key insight for the professor: L=20 beats L=50. The GRU already captures everything useful in the last 20 movies — looking further back adds noise, not signal. This is also a good real-world property: people's taste is driven by recent watching, not what they saw 5 years ago.

---

## Step 2 — E4: How big should the model be?

**Script to write:** `src/run_e4.py`  
**Command (once written):**
```bash
python -m src.run_e4
```

**What it does:**
Trains multiple model sizes (using best `max_len` from E3), varying two size parameters:
- `embed_dim` — how many numbers represent each movie (16, 32, 64)
- `rnn_units` — how many memory cells the GRU has (64, 128, 256)

Six combinations total (16×64, 16×128, 32×64, 32×128, 64×128, 64×256). Compares on
validation HR@10. Also tracks the train-vs-val loss gap to spot overfitting.

**What it contributes:**
- Answers: "did we just throw a big model at it, or did we choose the right size?"
- The professor will ask why you chose 128 units and embed_dim=32 — this gives you the empirical answer
- Overfitting evidence: if the gap between train and val loss grows with model size, bigger isn't better
- Lets you pick **smallest model within noise of the best** (key rubric point)

**Decision rule:** if d=32 / units=128 ties with d=64 / units=256, keep the smaller one.

- [x] `src/run_e4.py` written
- [x] E4 run completed
- [x] Best size noted: `embed_dim = 32`, `rnn_units = 64`

embed	units	HR@10	Gap	Verdict

16	64	0.2725	0.37	Too small, slow to converge (ep 44)

16	128	0.2704	0.44	More memory doesn't help tiny embeddings

32	64	0.2740	0.47	Decision winner — smallest within 0.005 of best

32	128	0.2779	0.52	Good but 2× the params of d32/u64 for +0.004 HR@10

64	128	0.2782	0.51	Raw best — but 4× params of d32/u64

64	256	0.2687	0.55	Clear overfitting — too big, stops at epoch 7



---

## Step 3 — E5: GRU or LSTM?

**Script to write:** `src/run_e5.py`  
**Command (once written):**
```bash
python -m src.run_e5
```

**What it does:**
Trains two identical models — one with a GRU cell, one with an LSTM cell — using the
best `max_len` and size from E3/E4. The `cell` parameter already exists in `model.py`,
so this is literally one argument change. Compares on validation HR@10 and training time.

**What it contributes:**
- Answers: "why GRU and not LSTM?" with an actual number, not just "we read it's faster"
- GRU has fewer parameters (no forget gate) so it's faster and less prone to overfit on smaller data
- Expected result: roughly a tie, or GRU slightly ahead — either way you have the evidence
- One slide: "GRU vs LSTM: GRU achieved HR@10=X vs LSTM HR@10=Y, and trains faster"

**Decision rule:** pick whichever has higher val HR@10. If tied within 0.3%, pick GRU (fewer params, faster).

- [ ] `src/run_e5.py` written
- [ ] E5 run completed
- [ ] Winner noted: `cell = "___"`

---

## Step 4 — Lock the config (critical — unblocks Lea and Cecile)

**What it does:**
Collect the winners from E3, E4, E5 and write them down in one place. Update the
constants at the top of `src/train.py`:

```python
MAX_LEN   = ___   # from E3
EMBED_DIM = ___   # from E4
RNN_UNITS = ___   # from E4
CELL      = "___" # from E5
```

Tell Lea the locked config so she can prepare the final test script.
Tell Cecile so she knows exactly what to train and export for the final artifact.

**What it contributes:**
- This is the moment the rest of the team can move
- Lea's official test run (on data never touched during tuning) uses this exact config
- Cecile's saved model uses this exact config
- Without this, everyone is training different things and results can't be compared

**Do NOT open the test set yourself.** Only Lea runs the official test, once, after this step.

- [ ] Winners from E3, E4, E5 collected
- [ ] `src/train.py` constants updated
- [ ] Lea notified of locked config
- [ ] Cecile notified of locked config

---

## Step 5 — Multi-seed validation (3+ seeds)

**Command:**
```bash
python -m src.train  # with SEED=42 (already there)
# then change SEED to 0, then to 7 in train.py (or pass as arg)
```

**What it does:**
Re-trains the locked config three times with different random seeds (42, 0, 7). Records
HR@10 and MRR for each run. Computes mean ± std across the three.

**What it contributes:**
- Proves the result is not a lucky random initialization
- The rubric requires ≥3 seeds for the final reported number
- The headline result becomes: "HR@10 = 0.273 ± 0.002 (mean ± std, 3 seeds)"
- This is what goes on Cecile's results slide

**Decision rule:** std should be small (< 0.005). If one seed gives wildly different results,
check if early stopping is kicking in too early.

- [ ] Seed 42 run logged
- [ ] Seed 0 run logged
- [ ] Seed 7 run logged
- [ ] Mean ± std HR@10 computed and noted: `___ ± ___`

---

## Step 6 — Know the model for the live Q&A

The professor asks about your code live. Below is every part of the model and what to say.

### The data pipeline (`src/data_prep.py`)
- Loads 1M ratings, sorts each user's history by timestamp
- **All-prefix training windows**: if a user watched 20 movies, we create 19 training examples (predict movie 2 from movie 1, predict movie 3 from movies 1-2, etc.) — this gives us ~160K training examples from 6K users
- **Leave-one-out split**: last movie = test, second-to-last = validation, rest = train
- Stats (popularity, min-count filter) computed on train data only — no leakage

### The model (`src/model.py`)
- **Embedding layer**: maps each movie ID to a 32-dimensional vector the model learns
- **GRU layer (128 units)**: reads the sequence of movie embeddings in order and compresses them into a single 128-number summary of "what this user likes"
- **Genre lookup (frozen)**: each movie also has a multi-hot genre vector (Action=1, Drama=1, etc.) appended to the embedding — frozen, not trained
- **Rating channel**: your star ratings (1–5, rescaled to 0–1) are appended as a third signal
- **Dropout (0.3)**: randomly zeroes 30% of neurons during training — prevents memorization
- **Softmax output**: outputs a probability for every one of the 3,706 movies — the highest probability is the recommendation

### Why GRU and not something else
- **vs. LSTM**: GRU is simpler (fewer parameters), trains faster, comparable accuracy on this data size (E5 proves it)
- **vs. Transformer**: Transformers need far more data and compute; GRU is appropriate for a 6K-user dataset
- **vs. just popularity**: GRU is personalized — it reads *your* history, not global trends (E0 proves it: 27% vs 4% hit rate)

### The evaluation (`src/evaluate.py`)
- **Full-catalog ranking**: for each user, rank all 3,706 movies and find where their actual next movie lands
- **HR@10**: did the right movie appear in the top 10? Averaged over all users
- **MRR**: the reciprocal of the rank — rewards getting it right at rank 1 more than rank 10
- **Why not sampled negatives**: sampling 100 random negatives inflates scores and makes comparisons misleading (E7 shows the gap)

---

## Summary Checklist

```
[ ] Step 0 — python -m src.train            → artifacts/ filled, Em/Cecile unblocked
[ ] Step 1 — E3 history sweep               → best max_len locked
[ ] Step 2 — E4 capacity sweep              → best embed_dim + rnn_units locked
[ ] Step 3 — E5 GRU vs LSTM                → cell type locked
[ ] Step 4 — Update train.py, notify team  → Lea + Cecile unblocked
[ ] Step 5 — 3-seed runs                   → mean ± std HR@10 for results slide
[ ] Step 6 — Know the model cold           → ready for live professor Q&A
```

> Scripts for E3, E4, E5 do not exist yet — ask Claude to generate them, or write them
> following the same pattern as `src/run_ablation.py`.
