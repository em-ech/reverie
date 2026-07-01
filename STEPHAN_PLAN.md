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
python -m src.gru_model.train
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

**Script:** `experiments/run_e3.py`  
**Command (once written):**
```bash
python -m experiments.run_e3
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

- [x] `experiments/run_e3.py` written
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

**Script:** `experiments/run_e4.py`  
**Command (once written):**
```bash
python -m experiments.run_e4
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

- [x] `experiments/run_e4.py` written
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

**Script:** `experiments/run_e5.py`  
**Command (once written):**
```bash
python -m experiments.run_e5
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

- [x] `experiments/run_e5.py` written
- [x] E5 run completed
- [x] Winner noted: `cell = "gru"` — tied with LSTM (diff=0.001), GRU wins on fewer params


cell       HR@10       MRR   NDCG@10  best_ep  s/epoch

------------------------------------------------------

GRU       0.2740    0.1345    0.1555       37       13 <-- best

LSTM      0.2730    0.1333    0.1544       33       13

---

## Step 4 — Lock the config (critical — unblocks Lea and Cecile)

**What it does:**
Collect the winners from E3, E4, E5 and write them down in one place. Update the
constants at the top of `src/train.py`:

```python
MAX_LEN   = 20   # from E3
EMBED_DIM = 32   # from E4
RNN_UNITS = 64   # from E4
CELL      = "gru" # from E5
```

Tell Lea the locked config so she can prepare the final test script.
Tell Cecile so she knows exactly what to train and export for the final artifact.

**What it contributes:**
- This is the moment the rest of the team can move
- Lea's official test run (on data never touched during tuning) uses this exact config
- Cecile's saved model uses this exact config
- Without this, everyone is training different things and results can't be compared

**Do NOT open the test set yourself.** Only Lea runs the official test, once, after this step.

- [x] Winners from E3, E4, E5 collected
- [x] `src/train.py` constants updated
- [ ] Lea notified of locked config
- [ ] Cecile notified of locked config

---

## Step 5 — Multi-seed runs (>=3 seeds) -> **LEA'S TASK**

### Lea's Full Guide — Final Test Run

**What she's doing:**
Training the locked model 3 times (different random seeds) on the test set — the data
that was never touched during tuning. The result is the official number for the
presentation slide.

---

### Why the test set is safe to use

All three splits — train, val, and test — come from the same `ratings.dat` file.
`build_dataset()` splits it automatically using leave-one-out:

```
User watched: Movie A -> Movie B -> Movie C -> Movie D -> Movie E

  Training data:   A, B, C        (used to train the model weights)
  Validation:      D              (used for early stopping + E3/E4/E5 decisions)
  Test:            E              (never touched — Lea opens this now)
```

During all of Stephan's E3/E4/E5 runs, the code only ever looked at `ds.val_target`.
The `ds.test_target` was sitting there the whole time but was never used. The model
never saw movie E, for any user, during training or tuning. `build_dataset()` enforces
this split every time it runs, the same way, deterministically.

---

### Step 1 — Get the code

**If she already has the repo cloned:**
```bash
git checkout master
git pull origin master
```

**If she's cloning for the first time:**
```bash
git clone https://github.com/em-ech/reverie.git
cd reverie
```

---

### Step 2 — Set up the Python environment

**Install uv (if she doesn't have it):**
```bash
# Mac / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Create the virtual environment and install dependencies:**

Mac/Linux:
```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install --native-tls -r requirements.txt
```

Windows (PowerShell):
```powershell
uv venv --python 3.12
.venv\Scripts\Activate.ps1
uv pip install --native-tls --link-mode=copy -r requirements.txt
```

> `--link-mode=copy` is only needed on Windows if the project is inside OneDrive.
> If not on OneDrive, she can drop it.

---

### Step 3 — Get the data

She needs `data/ml-1m/ratings.dat`. If it's not in the repo (large files are usually
gitignored):

```bash
# Download ml-1m dataset
curl -O https://files.grouplens.org/datasets/movielens/ml-1m.zip
# Mac/Linux:
unzip ml-1m.zip -d data/
# Windows (PowerShell):
Expand-Archive ml-1m.zip -DestinationPath data/
```

Check it's there:

```bash
ls data/ml-1m/
# should show: movies.dat  ratings.dat  users.dat
```

---

### Step 4 — Run the test evaluation

```bash
python -m src.gru_model.run_test
```

This will:

1. Load `ratings.dat` and build the train/val/test split automatically
2. Train the model 3 times (seeds 42, 0, 7) — each takes ~10 min on CPU
3. Evaluate on the test set after each run
4. Print the final number at the end

**Expected output at the end:**
```
==================================================
  FINAL TEST RESULTS (frozen config, test set)
==================================================
  HR@10  : 0.XXXX +/- 0.000X  (seeds [42, 0, 7])
  MRR    : 0.XXXX +/- 0.000X
  NDCG@10: 0.XXXX +/- 0.000X

  -> Slide number: HR@10 = 0.XXX +/- 0.00X (mean +/- std, 3 seeds, test set)
```

The last line is exactly what goes on the results slide.

---

### Step 5 — Push results back

After it finishes, the results are auto-logged to `results/log.csv`.
Lea should commit and push that:

```bash
git add results/log.csv
git commit -m "TEST results: 3-seed official test run"
git push origin master
```

---

### What "frozen config" means

The frozen config is the set of model settings decided by Stephan's experiments, already
hardcoded at the top of `src/run_test.py`:

```python
MAX_LEN   = 20    # decided by E3
EMBED_DIM = 32    # decided by E4
RNN_UNITS = 64    # decided by E4
CELL      = "gru" # decided by E5
```

**Do not touch these numbers.** The reason is a strict ML rule: you can only open the
test set once, with one config, decided before looking at the results.

If Lea runs the script, sees the result, thinks "that's low, let me try RNN_UNITS=128",
runs it again and gets a better number — that's cheating. She just used the test set to
tune the model, which makes it no different from validation. The professor will ask
"how did you pick your final config?" and the answer must be "we decided it on validation
data, then ran the test set once, blind." That's what makes the number trustworthy.

**Rule: run `src/run_test.py` exactly once, report whatever comes out.**

---

### Things Lea should NOT do

- Do NOT change `MAX_LEN`, `EMBED_DIM`, `RNN_UNITS`, or `CELL` in `src/run_test.py`
- Do NOT run `src/run_test.py` more than once — once it's run, the result is the result
- Do NOT run `src/train.py` or any E-script — those are Stephan's tuning scripts that use validation, not test

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
- **GRU layer (64 units)**: reads the sequence of movie embeddings in order and compresses them into a single 64-number summary of "what this user likes"
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

## Running the App — Execution Order & Reproducibility

The app needs 3 terminals running at the same time. Always start them in this order
(API must be up before the frontend tries to call it).

### Terminal 1 — Python scripts (your normal workspace)
```powershell
# Activate venv once per session (VSCode usually does this automatically)
.venv\Scripts\Activate.ps1

# Run any experiment script from here, e.g.:
python -m src.gru_model.train
python -m experiments.run_e3
```
**Needs venv:** YES — prompt shows `(reverie)` when active.

### Terminal 2 — API server (start first)
```powershell
.venv\Scripts\Activate.ps1
uvicorn app.api:app --port 8000
```
Wait for: `Application startup complete.`  (~30s — loads model weights)  
Check: open `http://localhost:8000/health` in browser → `{"status":"ok"}`  
**Needs venv:** YES (uvicorn lives inside `.venv`).  
**Keep open:** YES — closing it kills the API and breaks the frontend.

### Terminal 3 — Frontend (start after API is up)

**First time ever (one-time setup):**
```powershell
# If VSCode was already open when Node.js was installed, add it to PATH manually:
$env:PATH = "C:\Program Files\nodejs\;" + $env:PATH

cd "C:\Users\steve\OneDrive - IE University\Term 3\Deep Learning\netflix_sim\reverie\web"
npm install   # downloads node_modules/ — only needed once
npm run dev
```

**Every session after that:**
```powershell
# Only needed if "npm" is not found (i.e. VSCode was open before Node was installed):
$env:PATH = "C:\Program Files\nodejs\;" + $env:PATH

cd "C:\Users\steve\OneDrive - IE University\Term 3\Deep Learning\netflix_sim\reverie\web"
npm run dev   # skip npm install — node_modules/ already exists
```

> **Tip:** If `npm` is found without the PATH line, just skip it. It's only needed when
> VSCode was launched before Node.js was installed (PATH is stale for that session).

Wait for: `Local: http://localhost:5173/`  
Open that URL in your browser.  
**Needs venv:** NO — this is Node.js, completely separate from Python.  
**Keep open:** YES — closing it kills the frontend.

### Quick status check
| What | URL | Expected |
|---|---|---|
| API health | http://localhost:8000/health | `{"status":"ok","catalog_size":3413}` |
| Frontend | http://localhost:5173 | Reverie UI loads |

### Reproducibility note
All training runs are logged automatically to `results/log.csv`.
The locked config lives in `src/train.py` constants (top of file).
Anyone on the team can reproduce any result by checking out the repo,
running `uv pip install --native-tls --link-mode=copy -r requirements.txt`,
and running the same script with the same seed.

---

## What Lea Does Next

With the config locked, Lea's job is to run the **official final evaluation** —
the one number that goes on the results slide. She has two things to do:

### 1. Run the test set evaluation (≥3 seeds)
The test set has never been touched during E3/E4/E5. Lea opens it exactly once,
with the frozen config, and repeats 3+ times to get a stable mean ± std.

Lea needs a script (similar to `experiments/run_baselines.py` but using `ds.test_hist`
and `ds.test_target` instead of val). The locked config to use:
```python
MAX_LEN=20, EMBED_DIM=32, RNN_UNITS=64, CELL="gru"
# seeds: 42, 0, 7
```
Final reported number: `HR@10 = X.XXX ± 0.00X (mean ± std, 3 seeds, test set)`

### 2. Train-vs-val chart (the "not cheating" chart)
The loss curves are already saved by `track.save_history()` in `results/`.
Lea just needs to include `results/curve_artifact_full_hybrid.png` in the slide —
it shows train loss going down while val loss stays tracked, proving no overfitting.

---

## Summary Checklist

```
[x] Step 0 — python -m src.gru_model.train            → artifacts/ filled, Em/Cecile unblocked
[x] Step 1 — E3 history sweep               → max_len = 20
[x] Step 2 — E4 capacity sweep              → embed_dim = 32, rnn_units = 64
[x] Step 3 — E5 GRU vs LSTM                → cell = "gru"
[x] Step 4 — train.py updated, notify team → Lea + Cecile unblocked
[ ] Step 5 — LEA'S TASK: 3-seed test run   → mean ± std HR@10 for results slide
[ ] Step 6 — Know the model cold           → ready for live professor Q&A
```
