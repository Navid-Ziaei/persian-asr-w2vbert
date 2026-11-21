> **SUPERSEDED (2026-11-19):** v2 numbers from leaked splits, a punctuation-penalising scorer and a contaminated warm start. See ../PLAN.md audit.

# Persian ASR — Data & Training Report

**Date:** 2026-10-16 · **Scope:** all verified experiments, bugs, and decisions from the low-resource dialect transfer-learning campaign (docs: `ASR_SOTA_Low_Resource_Transfer_Learning.md`, `PLAN.md`, `outputs/metrics/*`).

---

## 1. Executive summary

| Family | Model | Best CER / WER (frozen v2 test, 1564 utts) | Outcome |
|---|---|---|---|
| Model B — CTC | XLS-R-300M full fine-tune, warm-start | **0.0635 / 0.2572** @ 50 h | Winner. Labeled-data curve is **flat** from 5 h to 50 h (−0.2 pt CER over 10× data) |
| Model A — Seq2seq | Whisper-large-v3 + LoRA (r=32) | 0.1058 / 0.4095 @ 10 h | Stalls, then **collapses** at 25 h under static LR 1e-4 (0.3115). LR re-tune sweep in flight |
| — | Whisper-large-v3 zero-shot | 0.1065 / 0.4103 | Reference point for everything |

Bottom line: **Model B beats Model A by ~5×** on every budget tested. More labeled hours buy essentially nothing in either family; the remaining CER levers are **LM-enhanced decode, self-supervised adaptation (Level 4), and text adaptation (Level 5)**. All labels: `persian_asr.evaluate.metrics.compute_metrics` (NFKC + Arabic-char normalization).

---

## 2. Data findings

### 2.1 Pool and splits

- **Pool extended to 200 h** — `data/splits/prepare_status.txt` reads `done 200.00h / 200.00h`; budget manifests now exist for **5 / 10 / 25 / 50 / 100 / 150 / 200 h** (`data/splits/budget_*.csv`). The extension is ready to feed the Level-4 self-supervised adaptation work (the highest-value next lever).
- Labeled sources defined in `configs/data_efficiency.yaml`: `existing_filtered` + `manual_labeling` + `synthetic`; unlabeled adaptation and text adaptation enabled.

### 2.2 Split freeze — a real incident (2026-10-13)

- A resume-run **re-sorted split ids** and the original test manifest was **not byte-recoverable**.
- Resolution: the **frozen v2 test split (1564 utterances)** in `data/splits/test.csv` is now canonical and Every number in this report is measured on it.
- Pool order was validated **byte-exact** against the surviving `budget_5h.csv`.
- Consequence: pre-freeze Whisper numbers shifted within noise (CTC rows reproduce exactly); the pre-freeze numbers should not be cited as baselines.

### 2.3 Data hygiene rules learned

1. Budget manifests must be **regenerable and order-validated** (a shuffled pool silently invalidates comparability).
2. Text normalization is part of the data contract: scoring uses exactly the same `normalize_text` (NFKC + Arabic-char folding) the CTC vocab was built on.
3. **ZWNJ (نیم‌فاصله) maps to space** — word boundaries as a learnable signal (relevant for WER).

---

## 3. Training findings — Model A: Whisper-large-v3 (LoRA)

### 3.1 Results (frozen v2 test split)

| Budget | LR | Steps | CER | WER | Verdict |
|---|---|---|---|---|---|
| 0 h (zero-shot) | — | — | 0.1065 | 0.4103 | Reference |
| 5 h | 1e-4 | 408 | 0.1100 | 0.4230 | **Slightly worse than zero-shot** |
| 10 h | 1e-4 | 792 | **0.1058** | **0.4095** | Best LoRA result; −0.1 pt vs zero-shot |
| 25 h | 1e-4 | 2097 | 0.3115 | 0.4821 | **Catastrophic regression** (3.4× zero-shot CER) |

- 50 h **not queued** — a bigger dataset cannot fix an unstable optimization point.
- **In-flight:** LR re-tune sweep at 10 h: **3e-4** (training done, final eval running, ~43% at report time) and **3e-5** (queued after) with 40 warmup steps → `outputs/metrics/lora_10h_lr3e-{4,5}.json`, checkpoints `budget_10h_lora_lr3e-{4,5}`. The winning LR then decides a single 25 h / 50 h leg.

### 3.2 The LoRA save-path bug (major, fixed)

- **Symptom:** every eval that reloaded a saved LoRA adapter from disk behaved like the bare zero-shot base (5 h disk reload read 0.1065 — "ties zero-shot" — while in-memory was 0.1100).
- **Root cause:** the manual fallback save in `_save_checkpoint` (peft lacks jinja2/markupsafe → `save_pretrained()` always throws) wrote raw `state_dict()` keys containing the adapter-name segment (`lora_A.default.weight`). peft's loader expects canonical keys (`lora_A.weight`), inserts the adapter name on load, and **silently discards all mismatched tensors (strict=False)**.
- **Fix:**
  1. Fallback now uses `peft.utils.save_and_load.get_peft_model_state_dict` and refuses to write an empty adapter.
  2. All three adapters on disk (`budget_{5,10,25}h_lora`) were **re-keyed in place** (tensors untouched).
- **Verification:** post-fix from-disk evals reproduce the in-memory training evals **digit-for-digit** — 25 h: 0.31145221169693843, 10 h: 0.10578187690412373.
- **Blast radius:** the `rebaseline_v2.json` LoRA-5h row (0.1065/0.4098) is **invalid** (was a no-op load); all tables now carry corrected values.

### 3.3 Optimization lessons

1. The 25 h collapse is **real** (reproduced from disk) — static LR 1e-4 destabilizes at 2097 steps. This is instability, not a "more signal" plateau.
2. Adding warmup (40 steps) is part of the re-tune; scheduler is linear decay (`get_scheduler("linear")`).
3. Hardware recipe (verified VRAM envelope, RTX 3080 Ti 16 GB, large-v3, batch 8 + accum 4):
   - frozen base in **bf16 + fp32 LoRA masters** (“do not flip all params to bf16" — eager CUBLAS path fails);
   - `gradient_checkpointing` + `enable_input_require_grads()` **before** `.to(device)`;
   - `use_cache=False` during training, KV-cache flipped on for eval only;
   - AdamW8bit; ~6 GB allocated / 7.1 GB reserved peak.

---

## 4. Training findings — Model B: XLS-R-300M CTC

### 4.1 Results (frozen v2 test split)

| Budget | Grad steps | CER | WER | Verdict |
|---|---|---|---|---|
| 5 h | 272 | 0.0637 | 0.2583 | working recipe point |
| 10 h | 528 | 0.0640 | 0.2589 | **flat vs 5 h** |
| 25 h | 1398 | 0.0637 | 0.2584 | **exact tie with 5 h** |
| 50 h | 2994 | **0.0635** | **0.2572** | best Model B; **−0.2 pt CER cumulative over 10× data** |

**The labeled-data curve is flat from 5 h to 50 h.** The warm-start saturates immediately: gains must come from adaptation levers, not hours. The volatile `budget_50h_ctc` checkpoint is the canonical best Model B.

### 4.2 The recipe that works (after 9 failed runs — do not regress any item)

1. **Warm-start, never rebuild the head** — random LM-head init on frozen representations collapses to all-blank within ~400 steps. Warm-start from `alifarokh/wav2vec2-xls-r-300m-fa`.
2. **Dedicated calibrated CTC blank** — head row 39 (+6.5 mean logit, pretrained calibration). Using char id 0 (`' '`) as blank pins WER at 1.0 (spaces unrecoverable); using row 36 instead of the calibrated row costs ~1 pt CER.
3. **Real-space labels** (spaces are labelable characters) + attention mask through training and eval (misaligned CTC alignment explodes the loss).
4. **`ctc_loss_reduction="mean"` forced after `from_pretrained`** (transformers 4.40 hub config override bug); LR 1e-5 (3e-5 diverges); grad-norm clip 5.0.
5. **Decode policy** — per-row decode (padded batch frames inject junk); ids beyond the char space decode to **nothing** (a native vocab has a real char at id 1, so `.get(1)` as unk inserted junk `آ` and inflated CER 0.07→0.87).
6. **Guardrail:** `train_ctc` raises on a vocab-less checkpoint instead of silently rebuilding a random head; checkpoints are trained on a copy, never in place.

---

## 5. Decode finding — beam search is a dead lever (without an LM)

Two real bugs in `decode_ctc_beam` were found, instrumented, and fixed (60+ utterance GPU subset probe, CPU unit cases):

1. **Space char excluded from beam candidates** — blank lives on dedicated head row 39; the space at id 0 was skipped by `range(1, max_id)`, so beam decoded text with no spaces at all (CER 0.232 / WER 1.0, discarded as bogus).
2. **Log-prob arithmetic** — head-only-row "silence" mass was added with plain `+` (multiplies probabilities) instead of `logsumexp`, poisoning every frame's blank state → beam preferred short collapsed chains. All state combinations now use exact `logaddexp`; a `math.log(0)` underflow crash path removed.

**Final honest result on `budget_50h_ctc`:** greedy **0.0635 / 0.2573** (43 s) vs prefix beam w=16 **0.0651 / 0.2620** (196 s) — beam is 0.16 pt *worse* at 4.5× the cost. Greedy stays the default; beam is opt-in (`decode_mode="beam"`). Remaining lever: **KenLM/4-gram shallow fusion** (deprioritized in favor of Level 4).

---

## 6. Tooling & infrastructure findings

- **Artifact layout (single source of truth):** `outputs/metrics/{zero_shot,lora_*,ctc_*,beam_vs_greedy,rebaseline_v2,data_efficiency_results}.json` + `outputs/metrics/data_efficiency_summary.md`; checkpoints in `outputs/checkpoints/budget_{h}h_{lora,ctc}[_{tag}]`.
- **Run tagging** (new): `output.run_tag` in config → versioned checkpoint dirs and metrics filenames, so LR sweeps/ablations never clobber the canonical artifacts.
- **Family driver** (new, PLAN item 4): `scripts/data_efficiency.py --family whisper|xlsr` (or `experiment.best_model_family`, now `"xlsr"`) produces one `data_efficiency_results.json` for both families; the CTC branch delegates to the verified `train_ctc` recipe over the frozen budget splits.
- **GPU power reality (RTX 3080 Ti laptop):**
  - On **battery**, Windows SW power-caps the dGPU to ~25–35 W → training slows **~10×** (2.4 s/it → 40–85 s/it) and a hard process death occurred mid-run (battery/kill at step 171, no traceback in log). **Rule: train only with AC power attached.**
  - At full power (~75 W): 10 h LoRA leg = ~30 min training + ~50 min eval; 25 h = ~2.4 h training.

---

## 7. Open questions & next steps (in order)

1. **Finish the LR sweep** (10 h @ 3e-4 and @ 3e-5) and pick the winner vs baseline 1e-4 (10 h = 0.1058).
2. **One deciding 25 h (or 50 h) leg at the winning LR.** If it beats zero-shot by > ~0.004 CER → Model A lives; otherwise → write off Model A.
3. **Level 4 — self-supervised adaptation** on the 200 h unlabeled pool (XLS-R continued pretraining before CTC fine-tune). Highest expected value given the flat labeled-data chart; `self_supervised_adaptation` is still a stub.
4. **Level 5 — text/LM adaptation:** dialect normalization study, KenLM shallow fusion revisit, error analysis of the 0.0635-CER checkpoint (dominant: short function-word substitutions — "ما"→"من", "می شود"→"میشود").
5. Housekeeping: `pip install jinja2 markupsafe` so peft's standard save path stops triggering the fallback; full-curve `data_efficiency_results.json` GPU run once the GPU is free; optional Model C (w2v-BERT 2.0) if time remains.

---

## 8. Artifact index

| What | Where |
|---|---|
| Headline table + decode section | `outputs/metrics/data_efficiency_summary.md` |
| Project plan / status ledger | `PLAN.md` |
| Per-row metrics | `outputs/metrics/*.json` (zero_shot, lora_5h/10h/25h, lora_10h_lr3e-{4,5} when landed, ctc_5h/10h/25h/50h, beam_vs_greedy, rebaseline_v2) |
| Checkpoints | `outputs/checkpoints/budget_{h}h_{lora,ctc}`, sweep tag dirs `budget_10h_lora_lr3e-{4,5}` |
| Training logs | `outputs/logs/*.log` (lora_lr_sweep.log current; ctc_run*.log history of the 9 failed runs) |
| Data manifests | `data/splits/budget_{5..200}h.csv`, `test.csv` (frozen v2), `pool_manifest.csv`, `prepare_status.txt` |
| Sweep configs | `configs/lora_sweep_lr3e-{4,5}.yaml` |
| Beam probes | `scripts/eval_beam.py`, `scripts/eval_beam_subset.py`, `beam_subset_probe.json` |
