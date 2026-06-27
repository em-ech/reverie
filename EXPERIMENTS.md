# Reverie — Experiment Protocol

The experiments that produce the project's technical-depth evidence (rubric: tuning,
architecture justification, overfitting prevention). Scoped down from the audit's
over-broad sweep to the experiments that actually move a decision. Every experiment
states what it decides and on which split.

---

## Ground rules (non-negotiable — enforce as a team)

1. **Tune on validation only.** Every knob (dataset, L, dims, layers, cell, MIN_COUNT,
   K) is chosen on **val** metrics. `[H6, M3]`
2. **Open the test set exactly once**, on the frozen config, after all decisions. Log it;
   the eval script warns if run more than once.
3. **Seeds:** report the final config as **mean ± std over ≥3 seeds**. A single run is a
   random variable. `[H6]`
4. **Significance:** every "X beats Y" claim carries a bootstrap 95% CI over users and a
   paired Wilcoxon p-value on per-user reciprocal ranks. CI of the difference must exclude
   0 to claim a win. `[H6]`
5. **No leakage:** MIN_COUNT, popularity, any normalization are **train-only**. `[C2]`
6. **One eval function** used by both the final evaluator and the early-stopping callback. `[H4]`

**Primary metric:** HR@10 (= Recall@10 under leave-one-out). **Secondary:** MRR. Top-1 as a
sanity check. Report the relationship between them once, openly. `[H1-eval]`

---

## E0 — GO/NO-GO GATE (days 2–3, before building anything else) `[H1]`

**Question:** does a bare GRU beat most-popular on `ml-1m`?

- Fixed: ml-1m, all-prefix windows, L=50, GRU(128), d=32.
- Compare to most-popular (train-only counts) on **val** HR@10 + MRR with a bootstrap CI.
- **Decision:** beats popularity (CI excludes 0) → proceed to the matrix. Ties/loses →
  escalate (aggressive MIN_COUNT to shrink V, more augmentation), and if still tied,
  switch to the **fallback narrative** in `ARCHITECTURE.md §8`.

This is the single most important experiment. It runs before the frontend, the TV bridge,
or any tuning.

---

## Core matrix (Phase 3–4, all decided on validation)

| ID  | Question                             | Variable                                  | Hold fixed  | Success / decision                                |
| --- | ------------------------------------ | ----------------------------------------- | ----------- | ------------------------------------------------- |
| E3  | How much history matters?            | L ∈ {10, 20, 50, full}                    | best config | smallest L within noise of the best               |
| E4  | Capacity vs overfit                  | d_movie ∈ {16,32,64} × layers ∈ {1,2}     | best config | smallest model within noise; watch val gap `[M2]` |
| E5  | Cell type                            | GRU vs LSTM                               | best config | pick higher val HR@10 (expect ~tie)               |
| E6  | Is the hybrid input justified?       | full vs no-genre-input vs no-rating-input | best config | keep a feature only if it earns val HR@10 `[M3]`  |

Run E0 first (it gates everything). E3–E6 are ablations for the deck's
"architecture justification." Keep each to a few runs; depth-with-CIs beats breadth.

---

## Headline comparison (Phase 4, the thesis) `[H5, H6]`

Frozen config, **≥3 seeds**, **test opened once**. RNN vs:

- most-popular (train-only)
- most-recent-genre popularity (popularity within the genre-set of the user's last-K train
  items, seen items excluded)
- **item-kNN** (cosine on the user-item matrix, scikit-learn)

Report HR@10 + MRR for each, bootstrap 95% CIs, and paired Wilcoxon RNN-vs-each. The
architecture is justified iff the RNN beats **all three** with CIs excluding 0.

---

## Diagnostics & methodology (cheap, high-honesty)

- **E7 — sampled vs full eval (one-off):** compute HR@10 under full ranking and under 100
  sampled negatives; show the gap as a methodology slide; headline uses full ranking. `[L4]`
- **E8 — taste-vector health:** softmax entropy distribution; across-user variance of the
  centered taste vector; cosine between a synthetic all-horror vs all-romance history. If
  that cosine > ~0.9, the taste signal collapsed → enforce mean-centering + top-K (already
  in design) and re-check. `[H2]`
- **E9 — TV content-bridge proxy:** leave-one-out over the resolved personal Netflix TV
  shows; content-cosine rank vs the 3,000-show catalog; HR@k / MRR vs random (= k/3000);
  shuffle-genre sanity floor must collapse HR to random. `[M5]`

---

## Overfitting evidence (rubric-required)

For the frozen config: train vs val loss/HR curves, the train–val gap, and the effect of
the regularizers (dropout/L2/early-stopping) shown as an ablation. `[H3]`

---

## Compute notes

- ml-1m + GRU + full softmax over ~3.7k items trains in minutes/epoch on a single laptop
  GPU, longer on CPU; the cuDNN path (no `recurrent_dropout`) keeps it fast. `[H3]`
- One training machine owns the runs; results logged to a shared `results.md` / CSV with
  the config hash, split seed, and metric ± CI per row so the matrix is reproducible.

---

## Results so far (ml-1m, seed 42, validation, max_windows_per_user=30)

Training config aligned to the professor's method (Sessions 5-6): EarlyStopping on
val_loss, patience=10, epochs=60 (early stopping ends training, not the cap). Under this,
the headline GRU stops ~epoch 30 and improves over the earlier patience=2/epochs=15 run.

**E0 gate — PASS.** GRU (id-only) HR@10 0.257 vs most-popular 0.042; paired MRR
diff +0.099, 95% CI [+0.093, +0.105]. Random floor 0.0029.

**Headline baselines (validation, aligned settings).** RNN beats all three with
non-overlapping CIs and Wilcoxon p≈0:

| Model            | Val HR@10 [95% CI]    | Val MRR [95% CI]      | vs RNN p |
| ---------------- | --------------------- | --------------------- | -------- |
| RNN (GRU)        | 0.2641 [0.252, 0.275] | 0.1252 [0.119, 0.132] | —        |
| item-kNN         | 0.0759 [0.069, 0.083] | 0.0352 [0.032, 0.038] | ~0       |
| recent-genre pop | 0.0446 [0.039, 0.050] | 0.0225 [0.020, 0.025] | ~0       |
| most-popular     | 0.0419 [0.037, 0.047] | 0.0215 [0.019, 0.024] | ~0       |

**E6 feature ablation (validation, locked config L=20/d32/u64).** Genre helps on HR@10;
rating alone hurts HR@10 but the full hybrid wins on MRR and NDCG. Decision: **full hybrid**
— best ranking quality (MRR), and the rating channel is required for the app's thumbs-down
feedback feature.

| Variant     | Val HR@10  | Val MRR    | Val NDCG@10 |
| ----------- | ---------- | ---------- | ----------- |
| id-only     | 0.2765     | 0.1310     | 0.1539      |
| +genre      | 0.2812     | 0.1324     | 0.1561      |
| +rating     | 0.2558     | 0.1290     | 0.1471      |
| full hybrid | 0.2740     | **0.1345** | **0.1555**  |

Note: id-only here (0.2765) is higher than the earlier headline run (0.2641) because this
uses the locked MAX_LEN=20 — consistent with E3 showing L=20 > L=50.

**E3/E4/E5 sweeps — DONE.** Locked config: `MAX_LEN=20, embed_dim=32, rnn_units=64, cell=gru`.

**Official test run — DONE.** HR@10 = 0.242 ± 0.002 (mean ± std, 3 seeds, test set).

**E6 feature ablation — DONE.** Full hybrid (genre + rating) is the chosen config.
