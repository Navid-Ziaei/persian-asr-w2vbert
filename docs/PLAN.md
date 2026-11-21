# Project Plan — Persian-ASR (low-resource dialect transfer learning)

**Status: audited 2026-11-19 (see Audit section; v2 results below are superseded)** — previously updated 2026-10-16 — supersedes the skeleton "Headline experiment" outline in `README.md` and tracks progress against `ASR_SOTA_Low_Resource_Transfer_Learning.md` §10–13 (Model A/B comparison → data-efficiency curve → adaptation levels).

## ⚠ Audit 2026-11-19 — read this before trusting any number below

A review of the plan, splits and scoring found problems that invalidate the headline conclusions ("data curve is flat", "XLS-R beats Whisper 5×"). All CER/WER tables below are **v2 numbers and are superseded**; they must be re-measured on v3 splits with the new scorer.

| # | Issue | Evidence | Status |
|---|---|---|---|
| 1 | **Sentence leakage**: all 1558 unique test sentences also occur in the 50 h train budget (different clips, same text). Flat 5→50 h curves and sub-7% CER are consistent with memorised sentences, not acoustic generalisation. | split audit | **Fixed** in `data/splits/v3/` (`scripts/make_clean_splits.py`: test kept frozen, train/dev stripped of any shared sentence, ≤5 clips/sentence). Only **78.6 h** of clean data remains, so 100/150/200 h budgets no longer exist. |
| 2 | **Scorer penalised Whisper for formatting**: `compute_metrics` kept punctuation (658/1564 refs contain ،؟ etc.) and ZWNJ; CTC heads cannot emit punctuation. Whisper-vs-CTC gap was partly an artefact. | `metrics.py` | **Fixed**: `normalize_for_scoring` (punct/diacritics stripped, digits unified, ZWNJ→space) on both sides. Whisper rows must be re-run. |
| 3 | **Warm-start checkpoint contamination**: `alifarokh/wav2vec2-xls-r-300m-fa` was already fine-tuned on Common Voice Persian; our "pool" is CV22 (`kiarashQ/farsi-asr-unified-cleaned`). The model has likely seen the test clips. | `configs/xlsr_300m.yaml`, `prepare_data.py` | **Open.** Rerun from the *base* `facebook/w2v-bert-2.0` / `xls-r` (needs a head recipe that doesn't collapse, see Roadmap) or report the warm-start arm as contaminated. |
| 4 | **Not a dialect study**: data is read-speech Common Voice; no dialect labels, no speaker ids. Title/claims overstate. | manifests | **Open**: either add real dialectal data (e.g. regional/conversational Persian sets) or rename to "low-resource Persian transfer". |
| 5 | **Non-nested budgets**: `budget_100h ⊄ budget_200h`, "200 h" is 196 h (pool re-sorted lexicographically after resume). Test = one contiguous CV id block. | audit | **Fixed** for v3 (seeded shuffle of id-sorted rows, nested prefixes, asserted). |
| 6 | `data/splits.py`: `build_budget_splits` used invalid `sorted(..., seed=)`; `get_data_efficiency_splits` kept only the one utterance crossing each threshold, so `scripts/data_efficiency.py` trained on **1 utterance**. | code | **Fixed**: replaced by `nested_budget_splits`, driver updated. |
| 7 | No predictions saved in metrics JSONs, so old runs cannot be re-scored. | `outputs/metrics/*.json` | Evaluate scripts must dump hyp/ref per utt. |
| 8 | Whisper LoRA "collapse at 25 h" and "no gain at 5/10 h" were never diagnosed (likely label/format mismatch from #2 plus LR/bf16); do not write Model A off on this evidence. | PLAN §Done | Re-run after #2. |
| 9 | Stale stack: transformers 4.40.2 (config-override bug workarounds), torch 2.3, peft 0.12. | `pip list` | Upgrade; drop workarounds that become unnecessary. |

### Work tracker (started 2026-11-19)

**A. Fixes**
- [x] A1 Sentence-leak-free nested v3 splits (`scripts/make_clean_splits.py`, `data/splits/v3/`; 5/10/25/50 h; test frozen)
- [x] A2 Scorer: punctuation/diacritics/ZWNJ-neutral (`normalize_for_scoring`) + tests
- [x] A3 Broken `splits.py` helpers replaced by `nested_budget_splits`; `data_efficiency.py` updated
- [x] A4 Per-utterance predictions dumped by every evaluator (`outputs/metrics/predictions_last_*.jsonl`) + `bootstrap_ci`
- [x] A5 `data.splits_subdir` honoured by all loaders (`resolve_splits_dir`); set `splits_subdir: v3` in configs
- [~] A6 (script ready: `scripts/eval_whisper.py zero_shot [--config configs/whisper_large_v3_turbo.yaml]`, now v3 + tagged outputs, never clobbers v2 files; GPU run pending) Re-baseline Whisper-large-v3 zero-shot on v3 + new scorer (GPU, ~45 min, AC power)
- [ ] A7 Upgrade transformers/peft, then drop the 4.40 workarounds
- [ ] A8 Resolve contamination (#3) and dialect-claim (#4) in the write-up

**B. Improvements (recent methods)**
- [x] B1 Model C: w2v-BERT 2.0 CTC from pretrained base (`train/w2vbert_ctc.py`, `scripts/train.py --mode w2vbert_ctc`; SpecAugment, warmup, adapter; CPU-tested tiny-model fwd/bwd + vocab/decode)
- [x] B2 First w2v-BERT probe (5 h, 1 epoch, ~14 min, v3 + new scorer): **test CER 0.0705 [0.068,0.073], WER 0.327 [0.315,0.336]**; dev 0.0861/0.364. Errors are mostly letter confusions (غ/ق, س/ص) and spacing variants (می شود/میشود) -> LM fusion (B5) and spacing-tolerant WER are the next levers. **3-epoch run: test CER 0.0490, WER 0.239** (dev 0.0637/0.271, still improving at epoch 3; `outputs/metrics/w2vbert_ctc_5h_3ep.json`). **6-epoch run (2 segments, resume verified): test CER 0.0435 [0.0413,0.0453], WER 0.2144 [0.2045,0.2235]** (dev 0.0541/0.235; `w2vbert_ctc_5h_ep6.json`). LM corpus built: `outputs/lm/fa.txt` (2.86M sentences from a 720 MB naab sample; 34 test/dev sentences found and removed -> naab does contain evaluation sentences, LM gains need a leakage caveat). Original command: `python scripts/train.py --config configs/w2v_bert2.yaml --mode w2vbert_ctc --budget_hours 5` (needs ~2.3 GB download, AC power)
- [ ] B3 Whisper large-v3-turbo zero-shot + LoRA/DoRA with LR sweep (config written)
- [ ] B4 Omnilingual ASR / MMS-1B zero-shot references
- [~] B5 (corpus fetcher `scripts/fetch_naab.py` tested on a 6 MB probe; full sample not yet downloaded; naab licence ambiguous MIT vs CC BY-NC-SA -> treat as non-commercial; code ready, untested end-to-end: needs `pip install pyctcdecode kenlm` + an ARPA built from external text) `scripts/build_lm.py` -> `scripts/eval_w2vbert_lm.py` (alpha/beta tuned on dev, bootstrap CIs). KenLM 4-gram + `pyctcdecode` shallow fusion (external Persian text)
- [ ] B6 Pseudo-labelling / noisy-student on unlabelled audio
- [ ] B7 Speed/noise augmentation
- [ ] B8 Final tables with bootstrap CIs

### Revised roadmap (recent methods, laptop-burn policy still applies)

1. **Re-baseline on v3 + new scorer** (≤45 min jobs): Whisper-large-v3 and **large-v3-turbo** zero-shot; save per-utt hyp/ref.
2. **Add Meta Omnilingual ASR (2025) / MMS-1B zero-shot** as strong multilingual references for Persian.
3. **Model C = w2v-BERT 2.0 CTC** (`Wav2Vec2BertForCTC`, SeamlessM4T-pretrained on 4.5M h, 143 languages) trained from the *pretrained base* with the standard HF recipe (LR ~5e-5 + warmup, SpecAugment time/feature masking on, char vocab built on `normalize_for_scoring` text, `ctc_zero_infinity=True`). It replaces the contaminated warm-start as the CTC arm. `train_ctc` is wav2vec2-only today — needs a family switch (not implemented yet).
4. **Whisper arm**: LoRA/DoRA r=32 on all linear layers, LR sweep {3e-4,1e-3} (LoRA wants higher LR than full FT), warmup, fp32 LoRA masters, labels without leading `<|startoftranscript|>` duplication, optionally decoder-only full FT; evaluate with the same scorer.
5. **LM fusion** (previously deprioritised, now the standard cheap win for CTC): KenLM 4-gram via `pyctcdecode`, LM trained on **external Persian text** (Wikipedia/OSCAR) minus test sentences — never on the train transcripts alone (small + risk of leakage).
6. **Level 4 replacement — pseudo-labelling / noisy-student** instead of continued pretraining: transcribe unlabelled audio with the best model + LM, filter by confidence, retrain. Continued SSL pretraining of a 300M–600M encoder is not feasible within the laptop-burn policy.
7. Augmentation: speed perturbation (0.9/1.0/1.1), noise/RIR, SpecAugment.
8. Report with bootstrap CIs over utterances (differences of 0.0002 CER, as in the "flat curve", are noise).

**Consolidated report:** everything in this plan is now also readable as a single narrative in `reports/data_and_training_report.md` (data + training + decode + infra findings, §1–8).

**Laptop-burn policy (user directive, 2026-10-16):** the machine is an unplugged/unstable-power RTX 3080 Ti laptop — long GPU jobs throttled on battery (~10× slower) and one mid-run process death happened. Rules going forward:
- **No multi-hour fire-and-forget chains.** Every GPU job must either (a) fit in < ~45 min, or (b) be split into short segments the user runs manually (checkpoint-resume ready).
- **Train only with AC power attached** — check `nvidia-smi --query-gpu=power.draw --format=csv,noheader` before/while running: ≥ ~70 W is healthy; ≤ ~35 W means throttled (don't bother starting).
- **Screen on subsets, confirm on full set.** 40–utterance probe → ~2 min; 250-utt stratified screen → ~10 min. Full 1564-utt evals (~45 min) are reserved for final numbers only.
- Resume support: at these budgets a killed run starts from scratch (minutes-scale per leg) — acceptable; checkpointing per epoch already exists (`partial` dirs).

**LoRA save-path bug (2026-10-16):** the peft fallback save in `_save_checkpoint` wrote raw `state_dict()` keys (`lora_A.default.weight`), which peft's loader silently discards on load (strict=False) — every reload from disk of a saved LoRA adapter was the bare zero-shot base. The **rebaseline_v2 LoRA-5h row (0.1065/0.4098, "ties zero-shot") is invalid**. Fix: fallback now writes canonical keys via `get_peft_model_state_dict`; the 5/10/25 h adapters were re-keyed in place, and from-disk evals now reproduce the in-memory training evals digit-for-digit. Real numbers below.

**Split freeze (2026-10-13):** a resume-run re-sorted split ids and the original test manifest is not byte-recoverable. The **frozen v2 test split (1564 utts)** in `data/splits/test.csv` is now canonical; pool order was restored and validated byte-exact against the surviving `budget_5h.csv`. All six original evaluations were re-baselined on it (`outputs/metrics/rebaseline_v2.json`): CTC rows reproduce exactly (0.0638/0.0640/0.0637/0.0635); Whisper zero-shot 0.1065/0.4103 and LoRA-5h 0.1065/0.4098 shifted within noise, rank order unchanged.

## Done (verified, all numbers on the same 1563-utterance held-out test set, `compute_metrics` NFKC normalization)

| # | Experiment | Data | CER | WER | Artifacts |
|---|---|---|---|---|---|
| 1 | Model A: Whisper-large-v3 zero-shot | 0 h | 0.1065 | 0.4103 | `outputs/metrics/rebaseline_v2.json` |
| 2 | Model A: Whisper-large-v3 + LoRA r=32 | 5 h | 0.1100 | 0.4230 | `outputs/metrics/lora_5h.json`, ckpt `budget_5h_lora` — slightly WORSE than zero-shot (re-baseline row was a no-op load, see bug note) |
| 7 | Model A: Whisper-large-v3 + LoRA r=32 | 10 h | 0.1058 | 0.4095 | `outputs/metrics/lora_10h.json`, ckpt `budget_10h_lora` — best LoRA result, still ~flat: 5× Model B gap |
| 9 | Model A: Whisper-large-v3 + LoRA r=32 | 25 h | 0.3115 | 0.4821 | `outputs/metrics/lora_25h.json`, ckpt `budget_25h_lora` — **catastrophic regression** at 1e-4 (3.4× zero-shot CER); LR re-tune required before any larger Model A run |
| 3 | Model B: XLS-R-300M CTC warm-start | 5 h | **0.0637** | **0.2583** | `outputs/metrics/ctc_5h.json`, ckpt `budget_5h_ctc` |
| 4 | Model B: XLS-R-300M CTC warm-start | 10 h | 0.0640 | 0.2589 | `outputs/metrics/ctc_10h.json`, ckpt `budget_10h_ctc` — **flat vs 5 h**: warm-start saturates, more labeled hours alone don't help |
| 5 | Model B: XLS-R-300M CTC warm-start | 25 h | 0.0637 | 0.2584 | `outputs/metrics/ctc_25h.json`, ckpt `budget_25h_ctc` — **exact tie with 5 h**: labeled-data curve is flat 5–25 h |
| 6 | Model B: XLS-R-300M CTC warm-start | 50 h | **0.0635** | **0.2572** | `outputs/metrics/ctc_50h.json`, ckpt `budget_50h_ctc` — new best Model B, but curve is flat: −0.2pt CER from 5→50 h (10× data) |
| 8 | Model B best ckpt, decode lever: prefix beam w=16 vs greedy | 50 h | 0.0651 vs **0.0635** | 0.2620 vs **0.2573** | `outputs/metrics/beam_vs_greedy.json` — **beam (no LM) is a dead lever**: 0.16 pt worse at 4.5× decode cost. Defaults stay greedy |

Summary table: `outputs/metrics/data_efficiency_summary.md`.

### Decisions locked in by experimentation (do not regress these)

The Model B arm went through 9 training runs; each collapse taught a durable lesson now encoded in `src/persian_asr/train/finetune.py` + `configs/xlsr_300m.yaml`:

1. **Warm-start, never rebuild the head.** A randomly re-initialized `lm_head` on frozen pretrained representations collapses to all-blank within ~400 steps at 5 h/1e-5. Warm-start from a language-matched community CTC checkpoint (`alifarokh/wav2vec2-xls-r-300m-fa`). Base XLS-R CTC from scratch at this budget is NOT trainable in practice.
2. **Dedicated calibrated CTC blank.** The 40-row head exposes a pretrained blank at head row 39 (+6.5 mean logit). Using char id 0 (`' '`) as blank makes real spaces unrecoverable (WER pinned at 1.0). Blank id = `old_out - 1`.
3. **Real-space labels** (ZWNJ → space) so word boundaries are learnable; decode keeps id 0 as a real char and skips only the dedicated blank row.
4. **`ctc_loss_reduction="mean"` forced after `from_pretrained`** (transformers 4.40 hub-config override bug), grad-norm clip 5.0, LR 1e-5 (3e-5 diverges), attention mask through training and eval.
5. **Decode policy**: per-row decode (padded frames of short rows inject junk in batched decode); ids beyond the char space decode to NOTHING — never map to `id_to_char.get(1)` (that was a real char in the native vocab; inflated CER 0.07→0.87).
6. **Guardrail**: `train_ctc` raises RuntimeError on a vocab-less checkpoint instead of silently rebuilding a random head; checkpoints are never trained in place (copy first).

What did **not** work: Whisper LoRA at 5 h is slightly worse than zero-shot (0.1100 vs 0.1065 CER, real post-fix number); 10 h improves −0.1pt; 25 h under the same 1e-4 schedule collapses to 0.3115 CER (digit-verified from disk vs in-memory). Not a flat-data curve but an unstable optimization at 1e-4 — re-tune LR (3e-5 vs 3e-4, warmup) before writing Model A off.

## In-flight / immediate next steps (order matters)

1. ~~Extend the Model B data-efficiency curve~~ **DONE**: 5/10/25/50 h all evaluated — CER 0.0637 → 0.0640 → 0.0637 → 0.0635. The labeled-data curve is **flat five-fold**: the budget-50h checkpoint is the best Model B for future work, and the flat curve **promotes LM-decode and self-supervised adaptation to the primary CER-reduction levers**.
2. **Match Model A at the same budgets** (Whisper LoRA @ 10/25/50 h): **10 h done** — 0.1058/0.4095 (`outputs/metrics/lora_10h.json`), still ~flat vs zero-shot. **25 h done** — caveats resolved (save-path bug fixed and re-keyed; in-memory 0.3115 reproduces from disk digit-for-digit): **catastrophic regression 0.3115/0.4821**. 50 h NOT queued: pointless before an LR re-tune. **LR re-tune sweep RUNNING** (`outputs/logs/lora_lr_sweep.log`): 10 h @ 3e-4 trained; its final eval ~50% at report time (battery-throttled, ~24 W — full-set eval is ~45 min on AC, potentially hours on battery); 3e-5 leg auto-starts after (short: ~30 min train on AC). **Decision shortcut given laptop constraints:** after leg 1's JSON lands (`outputs/metrics/lora_10h_lr3e-4.json`), compare against baseline 1e-4 = 0.1058 — if 3e-4 is already ≥ 1e-4 + 0.004 WORSE or collapses, 3e-5 remains the only candidate and the 3e-5 leg decides alone; run any further legs manually, one at a time, on AC power.
3. ~~LM-enhanced decode for Model B — beam search arm~~ **DONE, negative result**: clean prefix beam search (w=16, exact logaddexp sums, `decode_ctc_beam` in `finetune.py`) scores **0.0651/0.2620 vs greedy 0.0635/0.2573** on `budget_50h_ctc` — no gain without an external LM (beam only rescues low-mass boundaries, and here it preferentially chases short collapsed chains). Beam decode stays opt-in (`decode_mode="beam"`; `scripts/eval_beam.py`, `scripts/eval_beam_subset.py`). Remaining lever: **KenLM/4-gram shallow fusion** — deprioritized; adaptation (Level 4) leads. Two real decode bugs were found and fixed on the way (see summary `beam_vs_greedy` section): space char (id 0) excluded from beam candidates when `blank_id` is a dedicated head row; log-prob masses combined with `+` instead of `logsumexp`.
4. ~~Update `scripts/data_efficiency.py`~~ **DONE (code, 2026-10-16, full-curve GPU run pending)**: `run_data_efficiency` now dispatches on `experiment.best_model_family` (or `--family whisper|xlsr`). The `xlsr` branch delegates to the verified run-9 recipe (`finetune.train_ctc`) over the frozen budget splits and writes the shared sink `_write_metrics` → `outputs/metrics/data_efficiency_results.json` (family-tagged rows); the `whisper` branch keeps the LoRA loop, same sink. `configs/data_efficiency.yaml` flipped to `best_model_family: xlsr` (Housekeeping done).

## Later phases (per plan doc §10–12, still valid)

- **Level 4 — self-supervised adaptation** on unlabeled dialect audio: pool extension **DONE** (`data/splits/prepare_status.txt`: "done 200.00h / 200.00h"; `budget_150h.csv` + `budget_200h.csv` on disk) — continue XLS-R pretraining before CTC fine-tune. This is now the **highest-value next lever** given the flat labeled-data curve. (`self_supervised_adaptation` is still a NotImplementedError stub; when implemented, slice into short-segment GPU jobs per the burn policy above — e.g., sub-epoch chunks with checkpointing.)
- **Level 5 — text/LM adaptation**: dialect text normalization controlled study (نیم‌فاصله handling), spelling-variation analysis of the 0.0637-CER error distribution (dominant error: short function-word insertions/substitutions, e.g. "ما"→"من", "می شود"→"میشود").
- **Model C (w2v-BERT 2.0)**: optional third family if time remains after A/B curve completion — the Garhwali 2026 evidence makes it the strongest alternative encoder.
- **Model D**: best-of + adaptation stack (unlabeled + text + augmentation).

## Housekeeping

- `configs/data_efficiency.yaml` → `best_model_family: whisper` should flip to `xlsr` once the run-9 recipe is the default arm.
- LoRA config (`configs/whisper_large_v3.yaml`) needs an LR budget re-tune (1e-4 collapsed at 25 h; test 3e-4/3e-5 + warmup) before writing off Model A at bigger budgets.
- peft install lacks jinja2/markupsafe so `save_pretrained` always falls into the manual fallback — `pip install jinja2 markupsafe` (or keep the now-correct fallback; both paths produce canonical keys after the fix).
- Old training logs for the 9 CTC runs live in `outputs/logs/ctc_run*.log`; run-8 documents the regression that produced the vocab-less-checkpoint guard.
