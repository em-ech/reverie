# NextWatch — Experiment Protocol

The experiments that produce the project's technical-depth evidence (rubric: tuning,
architecture justification, overfitting prevention). Scoped down from the audit's
over-broad sweep to the experiments that actually move a decision. Every experiment
states what it decides and on which split.

---

## Ground rules (non-negotiable — enforce as a team)

1. **Tune on validation only.** Every knob (dataset, L, dims, layers, cell, MIN_COUNT,
   logit-adjustment, K, α) is chosen on **val** metrics. `[H6, M3]`
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

- Fixed: ml-1m, all-prefix windows, L=50, GRU(128), d=32, logit-adjustment ON.
- Compare to most-popular (train-only counts) on **val** HR@10 + MRR with a bootstrap CI.
- **Decision:** beats popularity (CI excludes 0) → proceed to the matrix. Ties/loses by
  day 3 → escalate (logit-adjustment, aggressive MIN_COUNT to shrink V, more augmentation),
  and if still tied, switch to the **fallback narrative** in `ARCHITECTURE.md §8`.

This is the single most important experiment. It runs before the frontend, the TV bridge,
or any tuning.

---

## Core matrix (Phase 3–4, all decided on validation)

| ID  | Question                             | Variable                                  | Hold fixed  | Success / decision                                |
| --- | ------------------------------------ | ----------------------------------------- | ----------- | ------------------------------------------------- |
| E1  | Does dataset size matter?            | ml-1m vs ml-latest-small                  | best config | pick the corpus with higher val HR@10             |
| E2  | Does logit-adjustment fix pop. bias? | logit-adj ON vs OFF                       | best config | ON if val HR@10 ↑ AND top-10 pop-bias ↓ `[M1]`    |
| E3  | How much history matters?            | L ∈ {10, 20, 50, full}                    | best config | smallest L within noise of the best               |
| E4  | Capacity vs overfit                  | d_movie ∈ {16,32,64} × layers ∈ {1,2}     | best config | smallest model within noise; watch val gap `[M2]` |
| E5  | Cell type                            | GRU vs LSTM                               | best config | pick higher val HR@10 (expect ~tie)               |
| E6  | Is the hybrid input justified?       | full vs no-genre-input vs no-rating-input | best config | keep a feature only if it earns val HR@10 `[M3]`  |

Run E0/E1/E2 first (they gate everything). E3–E6 are ablations for the deck's
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

## Results log (template — fill as runs complete)

| Exp | Config (hash) | Split seed | Val HR@10 | Val MRR | Notes |
| --- | ------------- | ---------- | --------- | ------- | ----- |
| E0  |               |            |           |         |       |
| ... |               |            |           |         |       |

Final (test, frozen config, mean±std over seeds):

| Model             | HR@10 (±) | MRR (±) | 95% CI vs RNN | Wilcoxon p |
| ----------------- | --------- | ------- | ------------- | ---------- |
| RNN               |           |         | —             | —          |
| most-popular      |           |         |               |            |
| most-recent-genre |           |         |               |            |
| item-kNN          |           |         |               |            |
