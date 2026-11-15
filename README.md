# ECG CNN-Transformer Classifier

A hybrid convolutional + Transformer deep learning model that classifies 12-lead ECG recordings into five cardiac conditions: **Normal**, **Atrial Fibrillation (AFIB)**, **Atrial Flutter (AFL)**, **Left Bundle Branch Block (LBBB)** and **Right Bundle Branch Block (RBBB)**.

The convolutional part learns local waveform morphology (P wave, QRS complex, T wave), while the Transformer part models long-range rhythm patterns across the full 10-second recording (RR-interval irregularity, loss of P waves). The repository covers the whole workflow: data loading and merging, training, an atrial-specific fine-tuning stage, evaluation, and saliency-based explainability.

## Why a hybrid model

ECG signals carry information at two scales:

- **Millisecond scale**: local wave shapes. Bundle branch blocks show up as widened, notched or amplitude-changed QRS complexes. A CNN captures these well.
- **Second scale**: rhythm. AFIB and AFL are defined by irregular or saw-tooth rhythm over many beats, which a CNN with a limited receptive field cannot capture on its own. Self-attention can relate any two time steps in the recording.

Pure CNNs lack global context for atrial rhythms, and hand-crafted-feature classical models lose information and cannot jointly model the 12 leads, so the model combines both ideas.

## Architecture

```
Input (B, 12, 5000)
  -> Shared-weight CNN encoder (applied to every lead independently)
       stem: Conv1d(k=7, s=2) -> Conv1d(k=3) -> MaxPool
       3 x MultiScaleResBlock (64 -> 128 -> 256 channels, stride 2)
            two parallel branches (kernel 7 and kernel 13), 1x1 fusion,
            Squeeze-and-Excitation channel attention, residual shortcut
  -> (B, 12, 256, 157)
  -> LeadTokenizer
       learned lead embedding + sinusoidal positional encoding + CLS token
  -> (B, 1885, 256)
  -> Transformer encoder (6 layers, 8 heads, GELU, pre-norm)
  -> CLS token (temporal features)  +  mean-pooled CNN features
  -> concatenate (512) -> Linear -> GELU -> Dropout -> Linear -> 5 logits
```

Design notes:

- **Weight sharing across leads** keeps the parameter count low (about 3.9M parameters) and reduces overfitting; lead identity is restored by the lead embedding.
- **Multi-scale residual blocks** let the network adapt to waves of different widths.
- **SE blocks** re-weight channels, which helps against noise such as muscle artifacts and baseline wander.
- **Aggressive temporal downsampling in the CNN** (5000 -> 157 steps per lead) keeps the Transformer sequence at 1,885 tokens. An earlier variant with 3,745 tokens was too memory-hungry because self-attention scales quadratically with sequence length.
- **Feature fusion** of the Transformer CLS token with pooled CNN features keeps both global rhythm and local morphology available to the classifier.

## Data

| Source | Content | Use |
| --- | --- | --- |
| Primary dataset | About 5,000 12-lead, 500 Hz, 10-second recordings in five roughly balanced classes (WFDB `.dat` + `.hea`) | All five classes |
| [PhysioNet large-scale 12-lead arrhythmia database](https://physionet.org/content/ecg-arrhythmia/) | 45,152 recordings, 12 leads, 500 Hz, 10 s, diagnoses as SNOMED-CT codes in the header | Extra AFIB and AFL records only (up to 1,000 per class) |

Why extra data only for AFIB and AFL: with the primary dataset alone, the Transformer did not have enough variety to learn rhythm irregularity, and the two atrial classes were the weakest. Adding atrial records only strengthens the Transformer without pushing the model toward majority classes.

SNOMED-CT mapping used for the PhysioNet records:

| Class | ID | SNOMED-CT code(s) |
| --- | --- | --- |
| Normal | 0 | 426783006, 427393009 |
| AFIB | 1 | 164889003 |
| AFL | 2 | 164890007 |
| LBBB | 3 | 164909002 |
| RBBB | 4 | 164934002 |

Only AFIB and AFL records are taken from PhysioNet; the rest of the mapping is kept for reference.

### Preprocessing and augmentation

- Fixed window of 5,000 samples (10 s at 500 Hz): longer records are center-cropped, shorter ones are zero-padded. Ten seconds covers enough beats to observe rhythm irregularity.
- Non-finite values replaced by zero; per-lead z-score normalization.
- Training-only augmentation: Gaussian noise, random time shift with zero filling, amplitude scaling. Validation and test data stay untouched.

## Handling class imbalance

- `WeightedRandomSampler` draws classes evenly in each batch.
- Hybrid loss: 70% class-weighted focal loss + 30% cross-entropy with label smoothing, so rare or hard classes get more focus without overconfident predictions.
- AFIB/AFL-focused **fine-tuning stage**: the CNN encoder is frozen and the remaining layers are trained for a few epochs at a low learning rate using only AFIB and AFL samples. The result is saved to a separate checkpoint so both models can be compared on the full five-class test set.
- Model selection uses validation **macro-F1**, with early stopping.

## Training setup

Defaults are defined in `ecg/config.py`:

| Setting | Value |
| --- | --- |
| Optimizer / schedule | AdamW, OneCycleLR (10% warm-up, cosine decay) |
| Learning rate / weight decay | 7e-4 / 1e-4 |
| Batch size / epochs / patience | 32 / 50 / 18 |
| Dropout | 0.2 (CNN), 0.1 (Transformer) |
| Focal gamma / label smoothing | 2.7 / 0.05 |
| Gradient clipping | 1.0 |
| Fine-tuning | 6 epochs, lr 5e-5, CNN frozen |

Multi-GPU training uses `DataParallel` automatically when more than one GPU is available. Checkpoints saved with or without the `module.` prefix are loaded transparently.

## Reference results

Obtained on a balanced held-out test set of 750 recordings (150 per class) from an earlier run with a different hyperparameter set (batch size 64, learning rate 1e-3, focal gamma 2.0, label smoothing 0.1). Results with the current defaults will differ.

Overall: accuracy 0.84, macro-F1 0.8407.

| Metric | Normal | AFIB | AFL | LBBB | RBBB |
| --- | --- | --- | --- | --- | --- |
| Precision | 0.9290 | 0.7059 | 0.6950 | 0.9577 | 0.9296 |
| Recall | 0.9600 | 0.8000 | 0.6533 | 0.9067 | 0.8800 |
| F1 | 0.9443 | 0.7500 | 0.6735 | 0.9315 | 0.9041 |

Normal, LBBB and RBBB are separated well (F1 above 0.90). The main error source is confusion between the two atrial rhythms: 27 AFIB recordings were predicted as AFL and 45 AFL recordings as AFIB. This is what the fine-tuning stage and the suggested cascade approach below target.

## Explainability

`scripts/evaluate.py` generates gradient-based saliency maps, one example per class, plotted over the lead with the highest saliency.

- Normal, LBBB and RBBB: attention concentrates on localized regions (for example the widened QRS in LBBB or the R wave in RBBB).
- AFIB: attention is spread across the whole recording, matching the global irregularity that defines the condition.
- AFL: attention often collapses onto one narrow window. In misclassified atrial examples, either irregular R-R intervals spread attention across the whole signal (AFL predicted as AFIB), or sudden high-amplitude isoelectric fluctuations resembled a saw-tooth pattern (AFIB predicted as AFL).

## Limitations and future work

- AFIB versus AFL remains the weakest point; rare bundle branch errors are mostly noise-related.
- A two-stage cascade is a natural next step: first separate atrial versus non-atrial, then use a dedicated model for AFIB versus AFL.
- Adding explicit rhythm features such as R-R interval statistics to the classifier input may further help the atrial distinction.

## Repository structure

```
.
├── requirements.txt
├── scripts/
│   ├── train.py
│   ├── finetune.py
│   ├── evaluate.py
│   └── run_all.py
└── ecg/
    ├── config.py
    ├── utils.py
    ├── losses.py
    ├── metrics.py
    ├── engine.py
    ├── checkpoint.py
    ├── training.py
    ├── finetune.py
    ├── testing.py
    ├── saliency.py
    ├── summary.py
    ├── data/
    │   ├── physionet.py
    │   ├── dataset.py
    │   └── builders.py
    └── models/
        ├── blocks.py
        ├── encoder.py
        ├── tokenizer.py
        └── classifier.py
```

## Getting started

```
pip install -r requirements.txt
```

Edit the dataset paths and hyperparameters in `ecg/config.py`. The primary dataset is expected to provide `train.csv`, `validation.csv` and `test_public.csv`, each with a `header_path` column (relative path to the WFDB header) and a `class_id` column (0 to 4). Run all commands from the repository root:

```
python -m scripts.train
python -m scripts.finetune
python -m scripts.evaluate
python -m scripts.evaluate --checkpoint best_model_afib_afl_finetuned.pt --no-saliency
python -m scripts.run_all
```

| Command | What it does |
| --- | --- |
| `train` | Prints the model summary, trains the main model, saves `best_model.pt` and `training_history.csv` |
| `finetune` | Loads `best_model.pt`, freezes the CNN, fine-tunes on AFIB/AFL, saves `best_model_afib_afl_finetuned.pt` |
| `evaluate` | Test metrics, per-class report and confusion matrix for any checkpoint, optional saliency maps |
| `run_all` | Full pipeline with a comparison of the original and fine-tuned models |

After fine-tuning, always compare both checkpoints on all five classes: updating the Transformer and classifier can affect the other classes as well.

## References

1. C. Che et al., "Constrained transformer network for ECG signal processing and arrhythmia classification," BMC Med. Inform. Decis. Mak., 2021.
2. M. F. Safdar, R. M. Nowak, P. Pałka, "Pre-processing techniques and artificial intelligence algorithms for electrocardiogram (ECG) signals analysis: A comprehensive review," Comput. Biol. Med., 2024.
3. V. Thota et al., "A lightweight CNN-attention-BiLSTM architecture for multi-class arrhythmia classification on standard and wearable ECGs," CISP-BMEI, 2025.
4. Y. R. Jin et al., "Multi-class 12-lead ECG automatic diagnosis based on a novel subdomain adaptive deep network," Sci. China Technol. Sci., 2022.
5. J. Cai et al., "Multi-ECGNet for ECG arrhythmia multi-label classification," IEEE Access, 2020.
