# Trustworthy Tomato Leaf Disease Detection — README

Companion document for `Melkamzer_Anteneh.ipynb`.
The notebook implements the full **diagnose → fix → measure → deploy** loop requested in
`Grad_CAM_Assignment.pdf`: train a 10-class tomato leaf disease CNN, use Grad-CAM to find a
repeatable failure, apply a fix motivated by that failure, measure the change honestly, and
ship an inference tool that reports prediction + heatmap + trust level.

---

## 1. Running it on Google Colab (intended path)

1. Upload `Melkamzer_Anteneh.ipynb` to [Colab](https://colab.research.google.com/).
2. **Runtime → Change runtime type → T4 GPU** → Save.
3. **Runtime → Run all** (or `Ctrl+F9`). No manual installs are needed.
4. Allow ~45–60 minutes: dataset download/prep (~10 min), three training runs (~25–30 min),
   Grad-CAM diagnostics and counterfactuals (~5–10 min), inference demos (~1 min).
5. The notebook writes `baseline.weights.h5`, `aug_only.weights.h5`, `final.weights.h5` in the
   working directory; all figures and tables are printed inline.

**Automatic configuration.** Cell 1 detects hardware and picks the run profile:

| Runtime | Resolution | Data | Epochs per model | Backend |
|---|---|---|---|---|
| Colab T4 GPU (detected) | 224×224 | all 11,000 images | 10 | TensorFlow/Keras 3 (preinstalled) |
| CPU fallback | 64×64 | stratified, 60 images/class | 4 | PyTorch backend if TF unavailable |
| `NOTEBOOK_FAST=1` (self-test) | 32×32 | 12 images/class | 2 | same as above |

The PyTorch branch exists **only** so the notebook can be verified locally on machines without
AVX support (TensorFlow 2.x cannot run on this CPU). Colab never touches it.

---

## 2. Assignment requirements → where they are satisfied

| PDF requirement | Notebook section | Concrete evidence |
|---|---|---|
| Train baseline CNN on 10-class tomato subset | §2 (cells 5–7) | `TomatoCNN(strong_augment=False, attn_lambda=0)`; 4 conv blocks, 10-class softmax |
| Report accuracy, confusion matrix, per-class F1 on held-out test set | §2 (cell 8) | `accuracy_score`, `classification_report`, normalized `confusion_matrix` |
| Grad-CAM identifies ≥1 specific, repeatable failure | §3 (cells 9–13) | Last-conv Grad-CAM overlays; outer-ring attribution statistic; focused small-spot failure grid |
| Targeted fix clearly motivated by the diagnosis | §4 (cells 14–16) | Strong background-invariant augmentation + Grad-CAM border-attention penalty in `train_step`; ablation model |
| Measure improvement quantitatively **and** qualitatively, with before/after heatmaps | §5 (cells 17–20) | Comparison table (accuracy, macro-F1, border-swap robustness, flip rate, centre-crop, border/centre mass), bar charts, final confusion matrix, side-by-side heatmaps on identical inputs |
| Inference tool: image → class + heatmap + trust level | §6 (cells 21–23) | `PlantDiseaseExplorer.explain()`; four demos including OOD texture and occluded leaf |
| 1–2 page markdown technical report at end | §7 (cell 25) | Seven subsections: objective/data, diagnosis, fix, results, tool, limitations/next steps, reproducibility |
| All cells run with outputs visible | Entire notebook | Cells are executed top-to-bottom; outputs (text, tables, figures) are embedded in the `.ipynb` |
| Filename = full name with underscores | `Melkamzer_Anteneh.ipynb` | ✔ |
| Keras (`keras.Model` to stack layers) | §2 (cell 6) | `class TomatoCNN(keras.Model)` with `keras.Sequential` conv blocks, `keras.ops` math |

---

## 3. Cell-by-cell guide

### §0 — Setup (cells 0–1)
- **Cell 0 (markdown):** title, problem framing, section roadmap, Colab instructions.
- **Cell 1 (code):**
  - imports (`os`, `urllib`, `pathlib`, numpy, matplotlib, PIL, pandas, sklearn);
  - `_ensure()` installs pandas/pyarrow/scikit-learn if a stripped environment lacks them;
  - backend selection: explicit `KERAS_BACKEND` env var wins, else TensorFlow if importable, else torch;
  - seed handling: `keras.utils.set_random_seed` on the TF path, `torch.manual_seed` on the torch path (calling the Keras seeder with TF installed but unusable can crash a non-AVX CPU);
  - hardware detection → `CFG` dictionary (image size, batch size, epochs, per-class cap, penalty λ);
  - constants: `S` (image size), `BORDER_FRAC = 0.18` (outer ring width used both as diagnostic and regularisation target).

### §1 — Data (cells 2–4)
- **Cell 2 (markdown):** dataset provenance (Hugging Face mirror `Project-AgML/tomato_leaf_disease`, 1,100 images/class) and split policy.
- **Cell 3 (code):**
  - `TOMATO` maps label ids to the 10 class names; `CLASSES = sorted(TOMATO.values())` defines folder order (the model's internal label space);
  - downloads `agml_tomato.parquet` if absent; decodes the `image` dict's `bytes` field;
  - `write_split()` materializes `tomato_data/{train,val,test}/<class>/*.jpg` with a per-class cap on CPU;
  - `train_test_split` twice → stratified **80/10/10** split (test data never used for training or recipe selection);
  - old split folders are removed first, so re-runs with a different config stay consistent;
  - creates two stress inputs in `tomato_data/ood/`: synthetic blurred noise (out-of-distribution) and a 50%-occluded leaf;
  - prints a per-split class-count table.
- **Cell 4 (code):** one representative training image per class (sanity check that downloads/labels are correct).

### §2 — Baseline (cells 5–8)
- **Cell 5 (markdown):** architecture description and the deliberately naive recipe (resize + flip only).
- **Cell 6 (code):**
  - `LeafBatches(keras.utils.PyDataset)`: decodes every JPEG once into a uint8 RAM cache, serves float32 batches. Backend-independent (unlike `image_dataset_from_directory`, which needs TF even on the torch backend). `on_epoch_end()` shuffles training indices; `get_image(ds, idx)` maps a global test index to one image.
  - `STRONG_AUG`: six Keras preprocessing layers (flip, translation, zoom, rotation, brightness, contrast; reflect fill). Used only by the fixed variants.
  - `conv_block(channels, stage)`: `Conv2D → BatchNormalization → ReLU → Conv2D → BatchNormalization → ReLU → MaxPooling2D`, stacked four times (32/64/128/256 channels). A post-training BN-recalibration step (cell 6) recomputes running statistics from the training split so inference-mode predictions are correct on every backend.
  - `class TomatoCNN(keras.Model)`: one class for all three variants. `call()` = [optional augmentation] → `conv_features()` (Rescaling + 4 blocks) → `classify()` (GAP → Dropout → Dense-10 softmax). Keeping `conv_features()` and `classify()` separate makes Grad-CAM identical for every variant.
  - `train_step()`: manual cross-entropy via `keras.ops` plus the optional Grad-CAM border penalty. Backend branch: `tf.GradientTape(persistent=True)` on TF; `torch.autograd.grad` on torch. The penalty uses the Grad-CAM of the **true** class from the same forward graph, so the optimiser receives an explicit "move attention off the border" signal.
  - `ResetEpochMetrics`, `compile_model`, `fit_model`, `to_np`, `predict_probs`, `plot_hist`: small helpers for metric resets, training, tensor conversion and plots.
- **Cell 7 (code):** instantiate/train/save baseline; print parameter count and learning curves.
- **Cell 8 (code):** `plot_cm()` and `metrics()`; print full per-class `classification_report` and normalized confusion matrix; store `res_base` (accuracy, macro-F1, probabilities, labels, predictions).

### §3 — Grad-CAM diagnosis (cells 9–13)
- **Cell 9 (markdown):** Grad-CAM formula $\alpha^k=\frac{1}{HW}\sum_{ij}\partial y^c/\partial A^k_{ij}$, and the two probes: attribution location + causal counterfactual.
- **Cell 10 (code):**
  - `make_grad_model()`: functional proxy `keras.Input → conv_features() → classify()` returning `[feature_maps, predictions]` (works on both backends; `Sequential.output` does not exist on torch).
  - `GradCAM.__call__()`: gradients of the target class score w.r.t. last-conv feature maps, global average, ReLU, bilinear resize to image size, min-max normalize; `overlay()` blends jet heatmap with the input.
- **Cell 11 (code):** `cam_grid()` — four correct and four wrong baseline predictions as image | heatmap | overlay triptychs; the Markdown/figure makes the spatial failure visible.
- **Cell 12 (code):**
  - `cam_mass()`: fraction of positive attribution in a binary mask; masks = outer 18% ring and central 50% box (reference values for uniform attention: 0.590 / 0.250, printed so the statistic cannot be misread);
  - `class_border_colors()` computes each class's mean border colour from training images;
  - `swap_border_np()` repaints the ring with another class's colour, `center_crop_batch()` removes it;
  - `counterfactual_acc()` measures accuracy and prediction flip rate on initially-correct images under each edit;
  - prints attribution statistics, counterfactual table/bar chart, and sets `BORDER_REF` = baseline 90th-percentile border mass (later used by the trust score).
- **Cell 13 (code):** focused failure grid for the two visually similar small-spot classes, annotated with confidence and border mass.
- **Cell 14 (markdown):** written diagnosis explaining what the evidence says and why accuracy alone hid it.

### §4 — Targeted fix (cells 15–16)
- **Cell 15 (markdown):** the two-part intervention and its causal motivation.
- **Cell 16 (code):** trains `aug_only` (ablation: augmentation, λ=0) and `final_model` (augmentation + λ·border mass); plots all three learning curves plus border mass vs. epoch during final training. Saving checkpoints allows later reuse.

### §5 — Measure (cells 17–20)
- **Cell 17 (markdown):** presentation-oriented transition ("what changed, and how we know").
- **Cell 18 (code):** builds `results` for all three models on the same test set; `cam_location_stats()` and `counterfactual_acc()` are re-run per model; assembles the `comparison` DataFrame (clean accuracy, macro-F1, border-swap accuracy, flip rate, centre-crop accuracy, border/centre attention).
- **Cell 19 (code):** three bar charts (including a uniform-attention reference line), final model's confusion matrix and per-class F1.
- **Cell 20 (code):** before/after Grad-CAM on the **same** images — prefers cases the baseline got wrong and the final model got right, with border mass annotated on each overlay.

### §6 — Inference tool (cells 21–23)
- **Cell (markdown):** explains trust = confidence + stability + border penalty and threshold logic.
- **Cell (code):**
  - `PlantDiseaseExplorer.__init__()` holds model, Grad-CAM and the baseline-calibrated border reference;
  - `_prepare()` accepts a path, PIL image or numpy array and normalizes size;
  - `_stability()` predicts under four ±6 px translations and an 82% centre crop, returning agreement fraction;
  - `explain()` returns a dict with `predicted_class`, `confidence`, `top3`, `heatmap`, `overlay`, `raw`, `stability`, `border_mass`, `trust`, `trust_score`, and optionally plots input / heatmap / overlay. Trust formula: `0.55·conf + 0.25·stability + 0.20·(1 − border/border_ref)`; HIGH ≥ 0.80, MEDIUM ≥ 0.62, else LOW.
- **Cell (code):** four demos — Healthy leaf, Septoria (ambiguous small-spot), OOD texture, occluded leaf.

### §7 — Digest + report (cells 24–25)
- **Cell 24 (code):** auto-generates a Markdown table from `comparison`, so reported numbers always match the actual run (including GPU vs CPU config) — no hard-coded results.
- **Cell 25 (markdown):** the required 1–2 page technical report: objective/data, diagnosis, motivated fix, results, inference tool/trust, limitations/next steps, reproducibility.

---

## 4. Design decisions worth being ready to explain

1. **Why a `keras.Model` subclass instead of `keras.Sequential` only?** One class parameterizes all three experiments and exposes `conv_features()`/`classify()`, so Grad-CAM and the attention penalty are literally the same code path for baseline and fixed models. Keras layers are still stacked with `keras.Sequential` blocks, as the brief requests.
2. **Why compute Grad-CAM inside `train_step`?** To turn an *explanation* into a *training signal*: the border-attention penalty should punish exactly the behaviour diagnosed by Grad-CAM, not a proxy for it.
3. **Why counterfactual probes?** Attention heatmaps alone are correlational. Repainting/cropping the border and observing prediction flips shows the model *causally* relies on background.
4. **Why an ablation?** Without the augmentation-only model you cannot tell whether the gain came from data augmentation, the attention penalty, or both.
5. **Why trust instead of raw softmax?** Confidence is not reliability: unstable or border-driven predictions are downgraded even when softmax is high.
6. **Uniform-attention reference (ring 0.590, centre 0.250):** these masks cover different areas; the meaningful signal is the *shift* relative to these values and between models, not the raw ring number alone.

---

## 5. Presentation guide (~7 minutes)

| Step | Say | Show |
|---|---|---|
| 1 Data | "Balanced 10-class dataset, strict 80/10/10 stratified split; test set untouched until evaluation." | class gallery, count table |
| 2 Baseline | "Standard VGG-style Keras CNN; strong clean score, but is it looking at the right thing?" | learning curves, confusion matrix, F1 report |
| 3 Diagnosis | "Grad-CAM repeatedly lights up leaf edges/background. Border repaint flips predictions — this is a causal shortcut, not decoration." | Grad-CAM triptychs, border histogram, counterfactual bars, small-spot grid |
| 4 Fix | "Because the shortcut is background/framing, augment framing/colour and penalize border attention directly." | training curves, falling border mass |
| 5 Measurement | "Same test set, same inputs, three models; report both accuracy and robustness — including the ablation so nothing is cherry-picked." | comparison table, before/after heatmaps |
| 6 Tool | "`explain(path)` predicts, explains, and refuses to call a shaky prediction HIGH trust." | four live demos |
| 7 Report | "Written up with limitations and next steps." | final markdown report |

---

## 6. Interpreting results honestly

The comparison table and digest are generated from the run itself. On the T4/full-data profile the
effects should be pronounced; on the small CPU fallback profile they are smaller and noisier — the
notebook always reports the numbers it actually measured. If a metric moves the "wrong" way, say so
and discuss why (limited epochs, tiny subset, hand-crafted border mask) rather than hiding it; the
ablation and uniform-attention reference lines are there to support that conversation.

---

## 7. Files in this folder

| File | Purpose |
|---|---|
| `Melkamzer_Anteneh.ipynb` | **Submission**: complete, executed notebook (baseline → diagnosis → fix → comparison → tool → report) |
| `README.md` | This document |
| `Grad_CAM_Assignment.pdf` | Assignment brief |
| `tomato_data/raw/agml_tomato.parquet` | Cached dataset (downloaded automatically if missing) |
| `tomato_data/{train,val,test,ood}/` | Split folders and stress images created by cell 3 |
| `*.weights.h5` | Checkpoints written by the training cells |
| `baseline.pt` | Legacy artefact from an earlier PyTorch prototype — **unused**, safe to delete |

---

## 8. Troubleshooting

- **"No module named keras" on Colab:** `pip install -q keras` in a scratch cell, then restart runtime; Colab's TF ships Keras 3 compatible with this notebook.
- **Hugging Face download is slow/offline:** the notebook uses `agml_tomato.parquet` if already present in `tomato_data/raw/`; that cached file is included here, so Colab can skip the download if you upload the folder too.
- **CPU fallback takes long:** set `NOTEBOOK_FAST=1` for a fast end-to-end smoke run; unset it for meaningful numbers.
- **Kernel dies locally:** ensure `KERAS_BACKEND=torch` is set before launching the kernel; this machine's CPU lacks AVX and cannot execute TensorFlow 2.x.

---

## 9. Limitations and next steps (also in the report)

Grad-CAM is a coarse single-layer approximation; the border mask is hand-crafted rather than a learned foreground mask; the trust weights are heuristic and should be calibrated on human-reviewed data; the model should be fine-tuned on field photos and given an explicit "not a leaf" class before real deployment.
