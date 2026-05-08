# Fine-Tuning SmolVLM for Science Visual Question Answering

**Course:** Deep Learning (NYU, Spring 2026)  
**Author:** Prabhjeet Singh (ps5351@nyu.edu)  
**Kaggle Competition:** Pixels to Predictions  
**Best Score:** 0.92555 accuracy (Solution 21)

## Overview

This repository contains all code for fine-tuning [SmolVLM-500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct) on the ScienceQA visual question answering task. The model answers multiple-choice science questions given an image (diagrams, charts, scientific figures) and question text.

The approach uses Low-Rank Adaptation with Weight-Decomposed LoRA (DoRA), training only 4.79M parameters (under 1% of the model). Key techniques that drove accuracy from 0.78068 to 0.92555 include balanced factorial augmentation (expanding 3,109 training samples to 40,000), hybrid test-time augmentation, and multi-epoch checkpoint ensembling.

All experiments were run on Google Colab.

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── best_solution/              # Best training + inference notebooks
│   ├── train_best.ipynb
│   └── inference_best.ipynb
├── solution1/                  # Baseline: vanilla LoRA fine-tune
│   ├── baseline_lora.ipynb
│   └── submission-0.78068.csv
├── solution2/                  # Logit scoring inference
│   ├── logit_scoring_inference.ipynb
│   └── submission-0.72635.csv
├── solution3/                  # Prompt ensembling + calibration
│   ├── prompt_ensembling_calibration.ipynb
│   └── submission-0.74647.csv
├── solution4/                  # All-linear LoRA r=8
│   ├── all_linear_lora_r8.ipynb
│   └── submission-0.76458.csv
├── solution5/                  # 768px resolution
│   ├── resolution_768px.ipynb
│   └── submission-0.79476.csv
├── solution6/                  # Metadata + choice shuffling
│   ├── metadata_choice_shuffling.ipynb
│   └── submission-0.79275.csv
├── solution7/                  # MLP-only LoRA
│   ├── mlp_only_lora.ipynb
│   └── submission-0.77464.csv
├── solution8/                  # DoRA
│   ├── dora.ipynb
│   └── submission-0.80080.csv
├── solution9/                  # Connector LoRA
│   ├── connector_lora.ipynb
│   └── submission-0.89134.csv
├── solution10/                 # DoRA + label smoothing + 3x shuffling
│   ├── label_smoothing_3x_shuffle.ipynb
│   └── submission-0.88128.csv
├── solution11/                 # 512px, batch=8, grad_accum=2
│   ├── native_512px.ipynb
│   └── submission-0.88329.csv
├── solution12/                 # MLP LoRA (down_proj), r=13, 5-choice oversample
│   ├── down_proj_5choice_oversample.ipynb
│   └── submission-0.88732.csv
├── solution13/                 # Balanced factorial augmentation (~5K/type)
│   ├── balanced_factorial_20k.ipynb
│   └── submission-0.89939.csv
├── solution14/                 # LoRA r=18, alpha=36, 6 epochs
│   ├── rank18_alpha36_6epochs.ipynb
│   └── submission-0.91549.csv
├── solution15/                 # Full TTA on solution 14 adapter (inference-only)
│   ├── full_tta.ipynb
│   └── submission-0.92354.csv
├── solution16/                 # Focal loss + 40K + hybrid TTA
│   ├── focal_loss_40k_hybrid_tta.ipynb
│   └── submission-0.91348.csv
├── solution17/                 # Metadata + captions + augmentation + new prompt (metadata+captions+new ending)
│   ├── metadata_captions_prompt.ipynb
│   └── submission-0.85714.csv
├── solution18/                 # Metadata in prompt
│   ├── metadata_in_prompt.ipynb
│   └── submission-0.87525.csv
├── solution19/                 # 40K dataset (10K per type)
│   ├── balanced_factorial_40k.ipynb
│   └── submission-0.91951.csv
├── solution20/                 # Hybrid TTA on solution 19 (inference-only)
│   ├── hybrid_tta.ipynb
│   └── submission-0.92354.csv
├── solution21/                 # Multi-epoch ensemble + hybrid TTA (BEST)
│   ├── multi_epoch_ensemble_hybrid_tta.ipynb
│   └── submission-0.92555.csv
└── solution22/                 # Curriculum learning
    ├── curriculum_learning.ipynb
    └── submission-0.90744.csv
```

Each `solutionN/` directory contains the Jupyter notebook and the Kaggle submission CSV. Checkpoint and adapter zips are excluded from the repo due to size (available on request).

## Results

| Solution | Description | Score |
|----------|-------------|-------|
| 1 | Baseline: vanilla LoRA (attention, r=16) | 0.78068 |
| 2 | Logit scoring inference | 0.72635 |
| 3 | Prompt ensembling + calibration | 0.74647 |
| 4 | All-linear LoRA r=8 | 0.76458 |
| 5 | 768px resolution | 0.79476 |
| 6 | Metadata + choice shuffling | 0.79275 |
| 7 | MLP-only LoRA | 0.77464 |
| 8 | DoRA | 0.80080 |
| 9 | Connector LoRA | 0.89134 |
| 10 | DoRA + label smoothing + 3x shuffling | 0.88128 |
| 11 | 512px, batch=8, grad_accum=2 | 0.88329 |
| 12 | MLP LoRA (down_proj), r=13, 5-choice oversample | 0.88732 |
| 13 | Balanced factorial augmentation (~5K/type) | 0.89939 |
| 14 | LoRA r=18, alpha=36, 6 epochs | 0.91549 |
| 15 | Full TTA on solution 14 adapter (inference-only) | 0.92354 |
| 16 | Focal loss + 40K + hybrid TTA | 0.91348 |
| 17 | Metadata + captions + augmentation + new prompt (metadata+captions+new prompt ending) | 0.85714 |
| 18 | Metadata in prompt | 0.87525 |
| 19 | 40K dataset (10K per type) | 0.91951 |
| 20 | Hybrid TTA on solution 19 (inference-only) | 0.92354 |
| **21** | **Multi-epoch ensemble + hybrid TTA** | **0.92555** |
| 22 | Curriculum learning | 0.90744 |

## Best Configuration (Solution 21)

Solution 21 combines the adapter trained in Solution 19 with a multi-epoch checkpoint ensemble and hybrid TTA at inference time.

**Training (Solution 19):**
- Base model: `HuggingFaceTB/SmolVLM-500M-Instruct`
- Adaptation: DoRA (Weight-Decomposed LoRA)
  - Rank: 18, Alpha: 36, Dropout: 0.05
  - Targets: `q_proj`, `k_proj`, `v_proj`, `o_proj`
  - Trainable parameters: 4.79M (< 5M budget)
- Image resolution: 512px (longest edge)
- Optimizer: AdamW, lr=2e-4, weight_decay=0.01
- Scheduler: Cosine with warmup (10% warmup ratio)
- Batch size: 8, gradient accumulation: 2 (effective batch = 16)
- Epochs: 6
- Label smoothing: 0.1
- Max sequence length: 1024
- Dataset: 40,000 samples (10K per choice type via balanced factorial augmentation)

**Inference (Solution 21):**
- Multi-epoch ensemble: average logits from epoch 3, 4, 5, 6 checkpoints
- Hybrid TTA: full permutation ensembling for 2/3-choice questions, identity-only for 4/5-choice questions
- Combined via logit averaging across epochs and permutations

## How to Reproduce

### Prerequisites

- Google Colab account
- Kaggle competition data: download from the "Pixels to Predictions" competition page
- Google Drive storage for data and checkpoints

### Setup

1. **Download the competition data** from Kaggle and upload the `pixels-to-predictions/` folder to your Google Drive root. The folder should contain:
   ```
   pixels-to-predictions/
   ├── train.csv
   ├── val.csv
   ├── test.csv
   └── images/
       └── images/
           ├── train/
           ├── val/
           └── test/
   ```

2. **Install dependencies** (each notebook runs this in Cell 0):
   ```bash
   pip install "transformers==4.47.0" accelerate peft bitsandbytes pillow tqdm pandas
   ```

### Training the Best Model (Solution 19)

1. Open `solution19/train_v19.ipynb` in Google Colab
2. Run Cell 0 to install dependencies, then **restart the runtime**
3. Run all remaining cells
4. Training takes approximately 3-4 hours
5. Per-epoch checkpoints are saved to Google Drive under `checkpoints_v19/`

### Generating the Best Submission (Solution 21)

1. Ensure you have the trained checkpoints from Solution 19 in `checkpoints_v19/` on Drive
2. Open `solution21/inference_multi_epoch_ensemble.ipynb` in Google Colab
3. Run Cell 0 to install dependencies, then **restart the runtime**
4. Run all remaining cells
5. The notebook:
   - Loads epoch 3, 4, 5, and 6 checkpoints
   - Applies hybrid TTA (full permutation ensembling for 2/3-choice, identity for 4/5-choice)
   - Averages logits across epochs and permutations
   - Outputs `submission.csv` with 0.92555 accuracy

### Running Other Experiments

Each `solutionN/` folder is self-contained. Open the notebook in Colab, install dependencies (Cell 0), restart runtime, and run all cells. Solutions 15, 20, and 21 are inference-only and require adapters from their parent training runs (solutions 14, 19, and 19 respectively).

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| transformers | 4.47.0 | SmolVLM model and processor |
| peft | latest | LoRA/DoRA adapter training |
| accelerate | latest | Model loading and device management |
| bitsandbytes | latest | Quantization support |
| torch | (Colab default) | PyTorch |
| pandas | latest | Data loading |
| pillow | latest | Image processing |
| tqdm | latest | Progress bars |
| numpy | latest | Numerical operations |

See `requirements.txt` for exact versions.

## Hardware

All experiments were run on **Google Colab**.

## Key Takeaways

1. **Connector LoRA was the single biggest jump** (solution 9): targeting the vision-language connector modules gave a +9 point accuracy gain over attention-only LoRA.
2. **Balanced factorial augmentation** (solution 13-14): permuting answer choices to equalize training samples across 2/3/4/5-choice types was critical for pushing past 0.90.
3. **Hybrid TTA outperforms uniform TTA**: full permutation ensembling helps 2/3-choice questions but hurts 4/5-choice questions due to noise from too many permutations.
4. **Multi-epoch ensembling** (solution 21): averaging logits across multiple epoch checkpoints (3, 4, 5, 6) smooths variance and adds +0.006 over the best single checkpoint.
5. **Changing everything at once fails** (solution 17): adding metadata, captions, and prompt changes simultaneously dropped accuracy by 6 points.
