> **SUPERSEDED (2026-11-19):** v2 numbers; see ../PLAN.md audit.

# Data-Efficiency Comparison — Persian ASR, frozen v2 test split (1564 utterances)

**Split note (2026-10-13):** a resume-run re-sorted split ids and the original test manifest was not byte-recoverable. The **frozen v2 test split** is now canonical; all rows below are **re-measured on it**. CTC rows reproduce the pre-freeze numbers exactly; the Whisper rows shifted within noise (rank order unchanged). Pool order was validated byte-exact against the surviving `budget_5h.csv`.

| # | Approach | Train data | CER | WER | Notes |
|---|----------|-----------|-----|-----|-------|
| 1 | Whisper-large-v3 zero-shot | 0 h | 0.1065 | 0.4103 | Prompted `transcribe`, language `fa`; no training (v2 re-baseline) |
| 2 | Whisper-large-v3 + LoRA (r=32) | 5 h | 0.1100 | 0.4230 | 2.0% params trainable (31.5 M); 408 steps @ 1e-4; slightly WORSE than zero-shot (the older 0.1065/0.4098 re-baseline row was a silently no-op load — see save-path note) |
| 7 | Whisper-large-v3 + LoRA (r=32) | 10 h | 0.1058 | 0.4095 | 792 steps @ 1e-4, ~1.7 h train; −0.1pt vs zero-shot — still flat, far off the CTC arm |
| 9 | Whisper-large-v3 + LoRA (r=32) | 25 h | 0.3115 | 0.4821 | 2097 steps @ 1e-4, ~2.4 h train; **catastrophic regression** (3.4× zero-shot CER). LR 1e-4 destabilizes at this scale — re-tune (3e-4/3e-5) before any larger run |
| 3 | XLS-R-300M CTC full fine-tune (warm-start) | 5 h | **0.0637** | **0.2583** | Warm-start from community Persian CTC ckpt `alifarokh/wav2vec2-xls-r-300m-fa`; native 36-char head, dedicated calibrated blank row 39, real-space labels; 272 grad steps @ 1e-5 + grad-clip 5.0 |
| 4 | XLS-R-300M CTC full fine-tune (warm-start) | 10 h | 0.0640 | 0.2589 | Same recipe; 528 grad steps. **Flat vs 5 h** — warm-start saturates: extra labeled hours alone add nothing at this operating point |
| 5 | XLS-R-300M CTC full fine-tune (warm-start) | 25 h | 0.0637 | 0.2584 | Same recipe; 1398 grad steps. **Exact tie with 5 h** — the labeled-data curve is completely flat 5→25 h for the warm-started arm |
| 6 | XLS-R-300M CTC full fine-tune (warm-start) | 50 h | **0.0635** | **0.2572** | Same recipe; 2994 grad steps. Flat: **−0.2pt CER cumulative from 5→50 h** — 10× the data for a marginal gain |

Sources: `rebaseline_v2.json`, `zero_shot.json`, `lora_5h.json`, `lora_10h.json`, `lora_25h.json`, `ctc_5h.json`, `ctc_10h.json`, `ctc_25h.json`, `ctc_50h.json` in this directory.

**LoRA save-path note (2026-10-16):** every peft fallback save in `_save_checkpoint` wrote raw `state_dict()` keys including the adapter-name segment (`lora_A.default.weight`). peft's loader expects canonical keys (`lora_A.weight`) and silently **discards all mismatched tensors on load** (strict=False), so any earlier eval that reloaded a LoRA adapter from disk was actually the bare zero-shot base — including the original `rebaseline_v2.json` LoRA-5h row (0.1065/0.4098, invalid). Fix: fallback now uses `peft.utils.save_and_load.get_peft_model_state_dict`, and the 5/10/25 h adapters on disk were re-keyed in place. Post-fix verification: from-disk evals reproduce the in-memory training evals digit-for-digit (25 h 0.31145221169693843, 10 h 0.10578187690412373).

## Decode lever (PLAN item 3, tested on `budget_50h_ctc` = best Model B)

| Decode | CER | WER | Decode time (idle GPU, row-by-row) |
|---|---|---|---|
| Greedy (frame argmax — evaluation default) | **0.0635** | **0.2573** | 43 s |
| Prefix beam search, width 16, no external LM | 0.0651 | 0.2620 | 196 s |

Source: `beam_vs_greedy.json`. **Pure acoustic beam (no LM) gives no CER gain** — it scored 0.16 pt *worse* than greedy at 4.5× the decode cost. The beam implementation had two real bugs found and fixed via `beam_vs_greedy_probe` unit cases + subset probe (see `scripts/eval_beam_subset.py`):

1. **Space char was excluded from beam candidates**: the checkpoint's blank lives on a dedicated head row (39) and the space char sits at id 0, which `range(1, max_id)` skipped — beam decoded text with **no spaces at all** (CER 0.232 / WER 1.0 reported, discarded).
2. **Log-prob arithmetic**: head-only-row silence mass was added with plain `+` to the blank log-prob (multiplying probabilities instead of summing), poisoning the per-frame blank state by tens of nats and collapsing beam output to short single-char chains.

Fixed: id-0 chars are candidates; silence mass is folded with `logsumexp`; all state combinations use exact `logaddexp` (also removes a `math.log(0)` underflow crash path). What remains true: as a Model B lever without an external LM, beam search is dead — gains must come from **KenLM/4-gram shallow fusion**, self-supervised adaptation, or text adaptation.

## Key findings

- **CTC on XLS-R wins at 5 h** with the correct recipe: warm-start from a language-matched checkpoint + keeping its native output head. Reachable in ~35 min of RTX 3080 Ti (16 GB) training.
- **Doubled labeled data (10 h) does not move the warm-started CTC model** (0.0640 vs 0.0637 + CER), 25 h ties 5 h exactly (0.0637/0.2584), and even 50 h only reaches 0.0635/0.2572 (−0.2pt CER over 10× the data). The labeled-data curve is **flat 5–50 h**: the budget 50h checkpoint becomes the new best Model B, but gains must come from adaptation levers (unlabeled self-supervised pretraining, LM-enhanced decode) rather than more labeled hours.
- **Whisper LoRA stalls, then collapses**: 5 h is slightly *worse* than zero-shot (0.1100 vs 0.1065 CER), 10 h moves −0.1pt (0.1058/0.4095), and 25 h catastrophically regresses to 0.3115/0.4821 under the same 1e-4 schedule. The label-data curve for Model A rises then implodes: more hours at this LR destabilize rather than help. Before any 25/50 h rerun, re-tune the LR (test 3e-5 and 3e-4) and lower-warmup; not a "needs more signal" flat curve but an unstable optimization at 1e-4.
- What made the CTC arm work (all were required — each alone left CER ≥ 0.47 or collapse):
  1. **Native vocab, never rebuild the head**: random reinit of the LM head on frozen-representation features collapses to all-blank within ~400 steps at 5 h scale.
  2. **Dedicated CTC blank on the calibrated extra head row** (row 39 of the 40-row head; +6.5 mean logit, pretrained calibration — verified by logit probe). Using char id 0 as blank made real spaces unrecoverable (WER pinned at 1.0); using the first extra row (36) instead of the calibrated one cost ~1 pt CER.
  3. **Real-space labels**: spaces are labelable characters; ZWNJ maps to space.
  4. **Attention mask through training and eval** so `input_lengths` reflect true audio length (misaligned CTC alignment otherwise explodes the loss).
  5. **`ctc_loss_reduction="mean"` forced after `from_pretrained`** (transformers 4.40 hub-config override bug) and grad-norm clip 5.0 against AdamW spikes.
  6. **Decode policy**: per-row decode; ids beyond the char space map to NOTHING (never to a char id) — a native vocab has a real char at id 1, so `id_to_char.get(1)` as universal `<unk>` inserted junk ` آ ` on every silence frame and inflated CER by 12×.
