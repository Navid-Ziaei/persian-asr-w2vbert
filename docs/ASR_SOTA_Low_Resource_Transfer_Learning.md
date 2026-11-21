f# State of the Art ASR Models for Low-Resource Languages, Accents, and Transfer Learning

**Research update:** September 2026

## Executive summary

For a low-resource Persian/dialect ASR project, the most relevant model families to benchmark are:

1. **Whisper large-v3** — strongest general-purpose multilingual baseline and very practical for fine-tuning.
2. **XLS-R** — one of the most important cross-lingual/self-supervised transfer-learning baselines, especially when unlabeled speech is available.
3. **MMS (Massively Multilingual Speech)** — specifically designed for very broad low-resource language coverage; useful when the target language is covered by MMS.
4. **SeamlessM4T v2** — strong multilingual foundation model with useful cross-lingual and text-only adaptation strategies.
5. **w2v-BERT 2.0 / large multilingual encoder models** — strong modern self-supervised representation-learning approach; particularly relevant for CTC-style fine-tuning.
6. **SraVaani 1.0** — a very recent 2026 example showing that language-region-specific pretraining can outperform generic multilingual models for many low-resource dialects/languages.
7. **BuzzASR** — a very recent 2026 result showing that language-specific Whisper fine-tuning, tokenizer adaptation, and text-only adaptation can substantially improve over vanilla Whisper.

For the proposed Persian dialect study, **Whisper-large-v3 + XLS-R/w2v-BERT-style encoder + a language-model/decoder-enhanced CTC system** would make the most informative comparison. The key research question should not be "which model is largest?", but **which pretrained model benefits most from each additional hour of dialectal labeled data and/or unlabeled speech**.

---

## 1. Recommended model shortlist

| Model | Architecture / approach | Main strength | Low-resource relevance | Transfer-learning relevance | Recommendation |
|---|---|---|---|---|---|
| **Whisper large-v3** | Encoder-decoder Transformer | Strong multilingual zero-shot ASR; robust to accents/noise | High | **Very high** | **Primary baseline** |
| **XLS-R 300M / 1B / 2B** | wav2vec 2.0-style self-supervised encoder + CTC | Cross-lingual representation learning | **Very high** | **Excellent** | **Primary transfer baseline** |
| **MMS** | Multilingual wav2vec 2.0 family | Extremely broad language coverage | **Excellent** | **Excellent** | Strong low-resource baseline if language is covered |
| **SeamlessM4T v2** | Multimodal multilingual encoder-decoder | Cross-lingual transfer + text/audio adaptation | High | **Excellent** | Useful advanced comparison |
| **w2v-BERT 2.0** | Self-supervised speech representation learning | Strong multilingual acoustic representations | High | **Excellent** | Strong modern encoder baseline |
| **SraVaani 1.0** | 430M FastConformer, TDT-CTC | Region-specific training for 65 Indian languages/dialects | Very high for Indic languages | High | Important evidence for region-specific pretraining |
| **BuzzASR** | Language-specific Whisper adaptation | Monolingual fine-tuning + tokenizer replacement + text-only adaptation | **Very high** | **Very high** | Important 2026 adaptation strategy |

---

# 2. Whisper

## Why it matters

Whisper is a multilingual encoder-decoder Transformer trained on a very large weakly supervised speech dataset. Its main advantage for low-resource research is that the model can often provide a useful starting point even when the target language/dialect has little labeled data.

### Relevant for accents and dialects

Whisper has strong robustness to speaker variation, noise, accents, and recording conditions. However, **zero-shot performance should not be treated as sufficient evidence of good low-resource performance**. Recent studies show that Whisper can still perform poorly on genuinely underrepresented languages.

For example, a 2025 Hakka study reported extremely poor Whisper performance in a low-resource setting, illustrating that multilingual pretraining does not guarantee good performance for every language.

### Fine-tuning

Whisper can be adapted using:

- Full supervised fine-tuning
- Parameter-efficient fine-tuning
- LoRA/adapter-style methods
- Data augmentation
- Text-only/domain adaptation
- Language-specific tokenizer adaptation

A 2024 study systematically explored Whisper fine-tuning strategies for low-resource ASR and confirms that adaptation is important when labeled data is scarce.

### Verdict

**Use Whisper-large-v3 as the main practical baseline.**

Do not compare only zero-shot Whisper against a newly trained model. Include **fine-tuned Whisper** because the research problem is transfer learning to dialectal speech.

Sources:

- OpenAI Whisper: https://github.com/openai/whisper
- Whisper paper: https://arxiv.org/abs/2212.04356
- Low-resource fine-tuning study: https://link.springer.com/article/10.1186/s13636-024-00349-3
- Hakka low-resource evaluation: https://aclanthology.org/2025.rocling-main.62/

---

# 3. XLS-R

## Why it matters

XLS-R is one of the most important models for a low-resource ASR research project.

It extends wav2vec 2.0-style self-supervised learning to multilingual speech. The published XLS-R models were trained on nearly **500,000 hours of speech across 128 languages**, with models up to 2B parameters.

The key idea is:

> Learn general speech representations from large amounts of unlabeled multilingual audio, then fine-tune using a relatively small amount of labeled target-language speech.

This is almost exactly the transfer-learning setting of a low-resource dialect project.

### Why XLS-R is especially useful for a research proposal

Unlike an end-to-end model such as Whisper, XLS-R makes the transfer-learning mechanism particularly explicit:

**multilingual unlabeled speech → pretrained acoustic representation → target-language/dialect fine-tuning**

This makes it highly suitable for studying the value of additional labeled data.

### Model sizes

Useful practical checkpoints include:

- XLS-R 300M
- XLS-R 1B
- XLS-R 2B

For a resource-constrained academic/industrial project, **XLS-R-300M is the most practical starting point**.

### Evidence

The original XLS-R work reported relative error-rate improvements of roughly **14–34% on average** over prior systems across several multilingual ASR benchmarks.

Source:

- XLS-R paper: https://arxiv.org/abs/2111.09296
- Hugging Face documentation: https://huggingface.co/docs/transformers/model_doc/xls_r

---

# 4. MMS — Massively Multilingual Speech

MMS is particularly relevant to extremely low-resource languages.

Meta's MMS project trained:

- wav2vec 2.0 models covering **1,406 languages**
- a multilingual ASR model covering **1,107 languages**
- language-identification models covering **4,017 languages**

The important idea is to increase language coverage dramatically while leveraging self-supervised learning.

The original MMS results showed that its multilingual ASR model could more than halve Whisper's WER on 54 FLEURS languages while using substantially less labeled data.

### Why it matters for dialect research

MMS demonstrates an important research hypothesis:

> **Cross-lingual pretraining can compensate for missing labeled data.**

However, coverage does not mean that every language/dialect receives equally good representations. A target dialect can still benefit from additional in-domain fine-tuning.

Source:

- Meta MMS overview: https://ai.meta.com/research/publications/scaling-speech-technology-to-1000-languages/
- MMS paper: https://arxiv.org/abs/2305.13516

---

# 5. SeamlessM4T / SeamlessM4T v2

SeamlessM4T is a large multilingual multimodal model supporting speech recognition, speech translation, text translation, and speech generation.

It was trained using approximately **1 million hours of open speech audio** for self-supervised speech representation learning.

### Why it is relevant

A particularly interesting direction is **combining multilingual pretraining with text-only adaptation**.

A 2024 study on parameter-efficient adaptation of SeamlessM4T found that:

- text-only adaptation can exploit unlabeled text
- parameter-efficient ASR fine-tuning can be combined with this
- cross-lingual transfer can produce up to **17% relative WER reduction** in an extremely low-resource setting, including a setting with no labeled speech

This is highly relevant if a dialect project has:

- lots of unlabeled speech
- little transcribed speech
- a large text corpus

### Verdict

SeamlessM4T is valuable as an **advanced transfer-learning comparison**, but it may be heavier and less straightforward than XLS-R or Whisper for a small industrial project.

Sources:

- SeamlessM4T paper: https://arxiv.org/abs/2308.11596
- Low-resource parameter-efficient adaptation: https://aclanthology.org/2024.mrl-1.13/

---

# 6. w2v-BERT 2.0

w2v-BERT-style models combine self-supervised speech representation learning with Transformer-based contextual modeling.

They are important because modern low-resource ASR is increasingly moving toward:

**large-scale self-supervised pretraining → lightweight supervised adaptation**

rather than training an ASR model from scratch.

A particularly important 2026 low-resource study on Garhwali found that **w2v-BERT 2.0 with standard CTC reached 47.0% WER across five seeds**, outperforming the larger MMS-1B model in that experimental setting.

Importantly, the study found that:

> model/pretraining design mattered more than simply increasing parameter count.

This is a useful methodological lesson for the proposed project.

Source:

- 2026 Garhwali study: https://arxiv.org/abs/2608.10670

---

# 7. SraVaani 1.0 — important 2026 development

SraVaani 1.0 is especially interesting because it addresses **languages and dialects rather than only major languages**.

It is a 430M-parameter FastConformer model with a hybrid TDT-CTC decoder.

It was:

1. pretrained on **31,255 hours** of speech,
2. trained using audio-image alignment with about **11 million audio-image pairs**,
3. fine-tuned on roughly **31,270 hours** of transcribed speech,
4. released for **65 Indian languages and dialects**.

The model explicitly includes many low-resource varieties such as:

- Garhwali
- Gondi
- Kokborok
- Mizo
- Rajasthani
- Tulu
- Wancho
- and many others.

### Why this matters

SraVaani provides evidence for an important strategy:

> **Domain/region-specific multilingual pretraining can be highly valuable for low-resource dialects.**

It is not directly usable for Persian, but it is a very useful reference architecture and experimental precedent.

Source:

- SraVaani paper: https://arxiv.org/abs/2608.08235
- Model page: https://vaani.iisc.ac.in/models/sravaani
- Hugging Face: https://huggingface.co/ARTPARK-IISc/SraVaani-1.0

---

# 8. BuzzASR — very recent 2026 result

BuzzASR is a September 2026 research result that is particularly relevant to the proposed methodology.

It takes Whisper and performs large-scale **language-specific adaptation** across 102 languages.

The adaptation includes:

- monolingual Whisper fine-tuning
- tokenizer replacement
- text-only fine-tuning
- data augmentation

The reported results are notable:

- outperforms Whisper-large-v3 on **77 of 102 languages**
- reduces CER by more than **2.8× on average**
- achieves state-of-the-art open-source CER on 27 languages in its evaluation
- tokenizer replacement substantially improves tokenization efficiency for many languages

### Why it matters for Persian dialects

This suggests a very practical research recipe:

**Whisper multilingual model  
→ language/dialect-specific tokenizer  
→ fine-tuning on dialectal speech  
→ text-only adaptation  
→ optional augmentation**

This is highly relevant if the target dialect has enough text but limited transcribed audio.

Source:

- BuzzASR paper: https://arxiv.org/abs/2609.09554

---

# 9. What the current research suggests for accents/dialects

The literature increasingly supports **adaptation rather than training from scratch**.

A useful hierarchy is:

### Level 1 — Zero-shot

Use the pretrained model directly.

**Data requirement:** 0 labeled hours

Useful for establishing a baseline, but usually insufficient for a serious dialect system.

### Level 2 — Fine-tuning

Fine-tune a pretrained multilingual model using dialectal transcriptions.

**Data requirement:** potentially a few hours → tens/hundreds of hours

This should be the main experimental setting.

### Level 3 — Parameter-efficient fine-tuning

Use adapters/LoRA or freeze most of the pretrained model.

Useful when:

- GPU resources are limited
- multiple dialect-specific models are required
- the base model is large

### Level 4 — Self-supervised adaptation

Continue representation pretraining on **unlabeled target-dialect audio** before supervised fine-tuning.

This is especially interesting for a project where several hundred hours of unlabeled audio are available.

### Level 5 — Text adaptation / language modeling

Use dialect-specific text to improve decoding/tokenization/language modeling.

This is particularly promising for:

- spelling variation
- dialect-specific vocabulary
- named entities
- colloquial expressions
- Persian spacing and نیم‌فاصله
- pronunciation differences that affect transcription

---

# 10. Recommended experimental design for Persian dialect ASR

For the proposed Persian dialect project, I would **not** try to benchmark every available foundation model.

A focused and defensible comparison would be:

## Model A — Whisper-large-v3

**Purpose:** strongest practical multilingual encoder-decoder baseline.

Experiments:

- zero-shot
- fine-tuning with 10h / 25h / 50h / 100h
- optional LoRA

## Model B — XLS-R-300M

**Purpose:** explicit cross-lingual self-supervised transfer baseline.

Experiments:

- pretrained model + CTC
- fine-tuning with the same labeled-data budgets
- optional language-model decoding

## Model C — w2v-BERT 2.0 / comparable multilingual encoder

**Purpose:** modern self-supervised encoder comparison.

Use the same target-domain data budget.

## Model D — strongest practical baseline + domain adaptation

Take the best of A/B/C and add:

- unlabeled dialect audio adaptation
- dialect text adaptation
- augmentation

This directly tests whether **unlabeled data provides additional value beyond labeled data**.

---

# 11. Most important experiment: data-efficiency curve

Instead of reporting only one final WER/CER, measure:

| Labeled dialect speech | CER | WER | Relative improvement | Cost |
|---:|---:|---:|---:|---:|
| 0 h | — | — | — | 0 |
| 5 h | — | — | — | Low |
| 10 h | — | — | — | Low |
| 25 h | — | — | — | Medium |
| 50 h | — | — | — | Medium |
| 100 h | — | — | — | High |

Then compare:

**Existing filtered data → unlabeled archive → manual labeling → synthetic data**

This produces a much more useful industrial result than simply reporting the best final WER.

The key research question becomes:

> **How much CER/WER reduction does each additional hour of each data source provide, and at what cost?**

---

# 12. Recommended ranking for the project

### Best overall practical choice

**Whisper-large-v3**

Best combination of:

- strong baseline
- multilingual pretraining
- robustness
- mature tooling
- straightforward fine-tuning

### Best research model for cross-lingual transfer

**XLS-R-300M**

Especially appropriate when the research question is explicitly about:

- transfer learning
- low-resource speech
- unlabeled speech
- cross-lingual representation learning

### Best extremely multilingual low-resource reference

**MMS**

Excellent reference for the "massive language coverage" approach.

### Best advanced adaptation direction

**SeamlessM4T v2**

Particularly interesting when combining:

- unlabeled text
- multilingual speech
- parameter-efficient adaptation

### Best evidence for dialect/region-specific pretraining

**SraVaani 1.0**

Not directly applicable to Persian, but highly relevant methodological evidence.

### Most interesting new adaptation strategy

**BuzzASR**

Especially relevant for:

- language-specific fine-tuning
- tokenizer replacement
- text-only adaptation
- very low-resource languages

---

# 13. Final recommendation

For a **3–4 month, resource-constrained Persian dialect ASR project**, the model scope should remain manageable.

### Recommended core comparison

**Whisper-large-v3 vs XLS-R-300M**

Then select the better model and perform:

**Target-dialect adaptation → data-efficiency study → unlabeled-data adaptation → final evaluation**

This is more scientifically meaningful and computationally feasible than attempting to train a new foundation ASR model.

The literature also suggests that **model size alone is not a reliable predictor of low-resource performance**. Recent low-resource results show that a smaller, well-pretrained encoder can outperform a larger multilingual model depending on the language, pretraining distribution, decoder, and adaptation strategy.

Therefore, the proposal should frame the contribution around:

> **efficient transfer learning and data efficiency for low-resource Persian dialect ASR**

rather than developing a new ASR architecture from scratch.

---

## Key references

1. Baevski et al. — *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations*  
   https://arxiv.org/abs/2006.11477

2. Babu et al. — *XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale*  
   https://arxiv.org/abs/2111.09296

3. Radford et al. — *Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)*  
   https://arxiv.org/abs/2212.04356

4. Pratap et al. — *Scaling Speech Technology to 1,000+ Languages (MMS)*  
   https://arxiv.org/abs/2305.13516

5. SeamlessM4T — *Massively Multilingual & Multimodal Machine Translation*  
   https://arxiv.org/abs/2308.11596

6. Gupta et al. — *Parameter-efficient Adaptation of Multilingual Multimodal Models for Low-resource ASR*  
   https://aclanthology.org/2024.mrl-1.13/

7. Liu et al. — *Exploration of Whisper fine-tuning strategies for low-resource ASR*  
   https://link.springer.com/article/10.1186/s13636-024-00349-3

8. Romney Robinson et al. — *JHU IWSLT 2024 Dialectal and Low-resource System Description*  
   https://aclanthology.org/2024.iwslt-1.19/

9. Pulikodan et al. — *SraVaani 1.0: Scaling Inclusive Speech Recognition for Indic Languages*  
   https://arxiv.org/abs/2608.08235

10. Singh et al. — *BuzzASR: A Swarm of 100+ Monolingual Speech Recognition Models*  
    https://arxiv.org/abs/2609.09554

11. Batra et al. — *Seeds Before Objectives: Rethinking Evaluation for Low-Resource Garhwali ASR*  
    https://arxiv.org/abs/2608.10670
