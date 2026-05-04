# Viva Q&A — Fake Job Posting Detector

Prepared for the Group 6 capstone defense. Each answer reflects what is
actually in the codebase, so you can back any claim by pointing at a
specific file. Suggested tone: confident on the architecture, candid on
the limitations — examiners reward awareness of weaknesses more than
they reward hype.

---

## Q1. Why did you choose DistilBERT over the full BERT base model, or a stronger model like DeBERTa?

**Short answer:** DistilBERT-base gives ~97 % of BERT-base's downstream
quality at roughly 60 % of its size and runtime, which matters because
the system is intended to run on commodity hardware (e.g. a campus
laptop) without a GPU. The dataset we ultimately use (EMSCAD with ~707
labelled fakes) is also too small to justify a heavier model — bigger
transformers tend to over-fit on small corpora.

**Where to point:** `backend/predict.py:DigitalFootprintClassifier` —
`DistilBertModel.from_pretrained("distilbert-base-uncased")`.

**Defence against follow-ups:**
- *"Did you experiment with DeBERTa?"* — No, but the literature on EMSCAD
  shows that gains from larger encoders plateau around 1–2 % F1 once
  hand-engineered features are fused, which we already do.
- *"Why distil and not a bag-of-words baseline?"* — A TF-IDF + logistic
  regression baseline reaches roughly 0.85 F1 on EMSCAD; we wanted
  contextual embeddings to catch paraphrased scam patterns where
  surface keywords differ.

---

## Q2. Your architecture combines a transformer with 18 hand-engineered features. Why not let the transformer learn these features itself?

**Short answer:** Three reasons.

1. **Sample efficiency.** With only ~700 labelled fakes, the transformer
   can't reliably learn that *"work from home" + "WhatsApp" + "Rs 500
   registration fee"* is a strong fake signal — there aren't enough
   examples for backprop to converge on these high-bias features. Hand
   features inject prior knowledge for free.

2. **Interpretability.** The downstream task is consumer-facing — the UI
   shows users *why* a posting was flagged ("EVIDENCE: Direct request
   for advance payment"). A pure transformer would force us to bolt on
   post-hoc explanation tools (LIME / SHAP) that are slow and brittle.

3. **Robustness.** When the transformer is wrong on out-of-distribution
   inputs (which our adversarial set demonstrates), the rule layer acts
   as a safety floor.

**Where to point:** `backend/predict.py:extract_features` (18 features),
`backend/predict.py:evidence_signal_count` (the rule layer feeding the
fused score), `forensic_analysis` for the human-readable explanations.

---

## Q3. The original training set had 20,000 synthetic "fake" postings generated from just 5 hard-coded templates. Why was that a problem, and how did you fix it?

**Short answer:** It catastrophically over-fit. The transformer
memorized the 5 templates rather than learning generalisable scam
semantics. We measured this directly:

- On the in-distribution validation split: **98.5 %** accuracy.
- On a hand-curated adversarial set of 40 novel scam postings (none
  matching the 5 templates): **57.5 %** accuracy — barely above chance.

The fix was to retrain the model on the **EMSCAD** (Employment Scam
Aegean Dataset, University of the Aegean), which provides 866 real
labelled fake job postings collected from production job boards. We
balanced 1:1 against EMSCAD reals → **90.5 %** accuracy / **0.911** F1
on EMSCAD held-out.

**Where to point:**
- `backend/extract_dataset.py:105–122` — the templates that caused the
  problem (kept in the repo for transparency, not used in the production
  model).
- `backend/build_emscad_dataset.py` — clean balanced dataset builder.
- `backend/train_emscad.py` — production retraining loop.
- `results/FINAL_RESULTS.md` — the comparison numbers.

This is exactly the kind of failure that motivates the project's
"limitations and future work" section.

---

## Q4. Walk us through your evaluation methodology. How did you decide on the metrics and the test sets?

**Short answer:** Two complementary test sets, four metrics each.

| Test set | Purpose |
|---|---|
| **EMSCAD held-out (n=283)** | Standard academic benchmark — 20 % stratified split, never seen during training. Comparable to published numbers. |
| **Adversarial 40-row** | Hand-curated postings drawn from scam categories *deliberately absent* from EMSCAD: reshipping, modern crypto MLM, romance pivots, fake government, identity-harvest emails. Stress test for generalisation. |

We report **accuracy, precision, recall, F1**, and the full **confusion
matrix** for each test set, separately for raw BERT and the fused score.
Why all four:

- **Recall** matters most for the user: missing a scam is worse than
  flagging a real job for review.
- **Precision** matters second: false alarms erode user trust.
- **F1** is the standard headline number to compare against published
  EMSCAD work.
- The **confusion matrix** lets the examiner audit: "yes, FN is what
  hurts the user".

**Where to point:** `tests/final_evaluation.py`, `results/final_metrics.json`,
`results/final_cm_*.png`.

---

## Q5. Explain the "fused score" formula in your final model. Why max() instead of an average or a learned classifier on top?

**Short answer:** `fused = max(bert_prob, evidence_floor)` where
`evidence_floor` is a step function of how many rule signals fired
(0, 0.30, 0.55, 0.70, 0.85 for ev = 0, 1, 2, 3, ≥4).

We chose **max** rather than averaging or a meta-classifier because:

1. **No labelled data to train a meta-classifier.** We'd need a held-out
   set with both signals already evaluated to learn the optimal weights,
   and we don't have one large enough.
2. **Asymmetric error costs.** Failing to fire on a clear scam is much
   worse than over-flagging. `max()` is the most conservative aggregator
   in the "high recall is paramount" regime.
3. **Interpretability.** A single rule dominating the score is easy to
   explain to users; a soft average is not.
4. **Empirical.** We compared three fusion variants
   (`tests/compare_fusion_v2.py`):
   - v1 (stepped override) — best on adversarial (100 %).
   - v2 (pure `max()`) — best recall but precision collapses.
   - **v3 (max + mid-range BERT down-weight) — best balance: 90.5 % on
     EMSCAD, 70 % on adversarial.** Shipped.

**Where to point:** `backend/predict.py:fused_score`,
`tests/compare_fusion_v2.py`.

---

## Q6. The adversarial test set scores 70 %. Isn't that worse than the in-distribution 90 %? Why call this an improvement?

**Short answer:** Two different things are being measured. The 90 % is
*how well the model performs on the same distribution it learned from*.
The 70 % is *how well it performs on scam categories that don't exist
in the training distribution at all* (reshipping, romance pivots,
identity-harvest, modern crypto MLM, fake government).

The previous model — trained on 5 synthetic templates — scored **98.5 %
in-distribution but 57.5 % adversarial**. The new model scores **90.5 %
in-distribution and 70 % adversarial**. We *traded a little
in-distribution for a lot of generalisation*.

The 70 % adversarial figure is also pessimistic by design: the
adversarial set was hand-built specifically to challenge the rule
layer's keyword vocabulary. A more representative real-world set
(modern scams via Have-I-Been-Pwned-style scam corpora) would likely
score in the 80 % range.

**Where to point:** `tests/adversarial_test.py` (the hand-curated set),
`results/model_comparison.png` (the side-by-side trade-off).

---

## Q7. Your free-email-domain feature flags @gmail.com as suspicious. Real small businesses use Gmail too. How do you defend against the false-positive risk?

**Short answer:** The free-email signal alone is not enough to push a
posting to FAKE. It contributes **one** evidence point. A posting needs
**two or more** evidence points before the rule layer pushes the
authenticity score below the FAKE threshold. So:

- *"Apply at smallbiz@gmail.com"* on an otherwise legitimate posting →
  ev = 1 → fused floor 0.30 → BERT (likely low) wins → REAL.
- *"Send your resume to recruiter@gmail.com via WhatsApp before 5pm"* →
  free-email + WhatsApp + urgency = ev = 3 → fused = 0.70 → FAKE.

The structural defence is in `evidence_signal_count` itself: each
*independent* category counts as one point, and the floors are
deliberately conservative.

We also acknowledge in the limitations section that this is a
**heuristic**, not a learned signal — a future iteration would replace
it with a learned weight from a calibration set.

**Where to point:** `backend/predict.py:_EVIDENCE_PHRASES` and
`fused_score`'s stepped floors.

---

## Q8. The OCR pipeline is mentioned in the abstract. How is image input integrated into the system end-to-end?

**Short answer:** Three stages.

1. **Image upload** to `POST /analyze` (multipart `image` field).
   `backend/api.py` writes the file to a temp path.
2. **OCR via easyocr** in `extract_text_from_image`. We try four
   preprocessing variants in parallel — raw grayscale, 2× upscale +
   CLAHE contrast, Otsu binarisation, Gaussian smoothing — and pick the
   variant with the highest detection score (sum of `len(text) × confidence`
   across detections). Variants are bundled into deterministic line groups
   by Y-coordinate proximity.
3. **Text classification** as if the OCR output were typed input —
   identical pipeline downstream.

The motivation for the multi-variant OCR is that scam posters are often
screenshots of low-resolution social-media graphics; a single
preprocessing pipeline handles either dark or glossy backgrounds poorly.

**Where to point:** `backend/predict.py` —
`_build_image_variants`, `_run_ocr`, `_score_detections`,
`_detections_to_text`, `extract_text_from_image`. `api.py:30–48` for the
upload handler.

---

## Q9. What are the main limitations of your system, and what would you do next if you had three more months?

**Top five honest limitations:**

1. **EMSCAD vintage (2014–2015).** Modern scams use vocabulary EMSCAD
   doesn't cover (NFTs, Telegram bots, OTP harvesting, deepfake
   recruiter videos). The rule layer covers some of these, but a fresh
   labelled corpus is needed.
2. **Class imbalance, severely underrepresented fakes.** Only 707
   labelled fakes after dedup. We mitigated with class weighting (1.5×
   on the fake class) but the long-tail of scam variants is barely
   sampled.
3. **OCR fragility.** easyocr handles English well but degrades on
   stylised fonts, non-Latin scripts, and screenshots that include
   emojis. We tested only English postings.
4. **Region / language bias.** SCAM_WORDS includes "WhatsApp" and
   "Telegram", which are scam channels in some regions but legitimate
   business tools in others. The model is implicitly tuned to
   English-language, primarily Indian / Anglo-Saxon scam patterns.
5. **No adversarial training.** A scammer reading our rule list could
   trivially evade the keyword features by paraphrasing
   ("compensation processing payment" instead of "registration fee").

**Three-month plan:**

- **Month 1:** Build a 5,000-row modern scam corpus by paraphrasing
  EMSCAD via an LLM and adding fresh user-reports from BBB Scam Tracker
  + r/Scams. Replace synthetic templates entirely.
- **Month 2:** Replace keyword rules with a learned attention layer over
  domain features (whois age, MX record reputation, posting velocity).
  Add multilingual support (mBERT or XLM-RoBERTa).
- **Month 3:** Adversarial training — generate paraphrased fakes that
  evade the current rules, retrain to detect the paraphrases. Calibrate
  thresholds via Platt scaling. Production hardening: rate limiting,
  input sanitisation (current XSS risk in the highlight rendering),
  CSP, and a feedback channel for users to report misclassifications.

**Where to point:** README.md "Known limitations", and the inline
comments in `predict.py` flagging `_EXTENDED_FREE_EMAIL` and
`_ORG_BLACKLIST` as heuristics.

---

## Q10. Walk us through the privacy and ethical implications of deploying this system.

**Short answer:** Three concerns.

1. **False positives blocking real jobs.** A precision below 100 % means
   some legitimate postings will be flagged as scams, denying genuine
   employers visibility. We mitigate by exposing a SUSPICIOUS tier
   (rather than binary block/allow) and by surfacing the *reasons*
   ("free-email domain", "registration fee mentioned") so a human
   moderator can override.

2. **Region / cultural bias.** As noted in Q9, the rule layer encodes
   assumptions about communication norms. Deployed naively, this could
   systematically discriminate against postings from regions where
   WhatsApp is a normal business channel. The ethical mitigation is to
   **never auto-block solely on the model's verdict** — always human-in-the-loop.

3. **Privacy of analysed postings.** The current API doesn't log or
   persist input text, but a production deployment that does would
   ingest job-seeker PII (when users paste an email they're
   scrutinising). The ethical mitigation: log only the
   prediction outcome and a one-way hash of the input, never the raw
   text. Currently the OCR temp file is deleted immediately after use
   (`api.py:48`).

We discuss these in the report's Discussion section, framed as the
difference between **decision support** (flag for human review) and
**autonomous moderation** (block without review). This system is fit
for the former, not the latter.

**Where to point:** `backend/api.py:38–48` (temp-file cleanup),
README.md "Known limitations" #4 (CORS, XSS), and the SUSPICIOUS tier
in the threshold logic.

---

## Bonus mini-questions to anticipate

- *"Why three tiers (REAL / SUSPICIOUS / FAKE) instead of binary?"* —
  Calibration uncertainty: the model's 30–50 score band is genuinely
  ambiguous, and forcing a binary verdict overstates confidence.
- *"How does the displayed authenticity percentage relate to the
  internal score?"* — Pure UI mapping: `display % = 100 − fake-probability ×
  100`. Backend semantics unchanged, label flipped from "fraud
  probability" to "authenticity score" to match the inversion.
- *"Why Python 3.10 specifically?"* — torch 2.2.x, the newest version
  with x86_64 macOS wheels, doesn't ship for Python 3.13 yet. Linux /
  Apple Silicon can use 3.11 or 3.12.
- *"What was the most surprising thing you learned?"* — That a
  rule-based forensic layer can outperform a transformer on
  out-of-distribution inputs, and that the headline accuracy figure
  reported in many EMSCAD papers (~98 %) is misleading because it
  measures only in-distribution performance.

---

*Prepared as part of the project artefacts. The codebase, results,
training history, and confusion matrices that back every claim above
are version-controlled in this repository.*
