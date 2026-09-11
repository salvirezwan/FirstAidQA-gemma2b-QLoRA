# HandsOnAid: Building and Evaluating an Offline First-Aid Assistant for Emergency Response Scenarios

This repository contains the code, data, and fine-tuned models for **HandsOnAid**, an offline AI first-aid assistant designed to run locally on resource-constrained consumer devices during emergency scenarios. 

We fine-tuned a 4-bit Gemma 2B model using QLoRA on the synthetic FirstAidQA dataset and rigorously evaluated it against strict medical guidelines (ANZCOR) using four blind LLM judges. While fine-tuning significantly improved performance, we found that the evaluation scores were ultimately capped by the absence of highly specific clinical protocols in the training corpus, highlighting the critical importance of strict dataset-guideline alignment in medical AI.

---

## Repository Structure

```
├── data/                                    # Raw and enriched datasets (FirstAidQA)
├── splits/                                  # Fixed train, validation, and test splits
├── experiments/                             # Pre-trained LoRA adapter weights for evaluation
├── judging/results/                         # Final aggregated scores and full LLM judgments
├── utils/                                   # Helper scripts (downloading models, diagnostics)
├── data_v2.py                               # Dataset preparation script
├── train_v2.py                              # Main QLoRA training script
└── eval_suite.py                            # Evaluation script for generating answers
```

---

## Requirements & Installation

We recommend using a conda environment to manage dependencies. Training requires a GPU with at least 12GB of VRAM (for 4-bit QLoRA). Inference can be run on CPU or a consumer-grade GPU.

```bash
# 1. Create the environment
conda create -n fine_tuning python=3.11 -y
conda activate fine_tuning
pip install -r requirements.txt

# 2. Cache the base Gemma 2B model locally (Recommended)
# Ensure you have accepted the Gemma terms at huggingface.co/google/gemma-2b-it and logged in via `huggingface-cli login`
python utils/download_model.py
```

---

## Reproducing the Experiments

This repository contains everything needed to reproduce the 7 model configurations (A through G) and the evaluation tables presented in the paper. 

### 1. Training (Fine-tuning)
To reproduce the fine-tuning of the primary 4-bit QLoRA adapter on the 10-category split:

```bash
python train_v2.py \
  --quant 4bit \
  --model_path models/gemma-2b-it \
  --splits_dir splits/10cat \
  --splits_tag 10cat \
  --lora_r 16 --lora_alpha 32 \
  --lr 1e-4 --patience 3 --seed 42
```

### 2. Generating Model Answers (Inference)
The pre-trained LoRA adapters are provided in the `experiments/` directory. To generate answers for the 41-question evaluation bank across all configurations:

```bash
python eval_suite.py
```
This will output the raw generated answers to a new folder in `evaluations/`.

### 3. Reviewing the Results
All final aggregated results, statistics, and full generated answers from the LLM judges (DeepSeek, Claude, GPT, GLM) used to populate the tables in the paper are pre-computed and stored in:
`judging/results/`

You can inspect the `stats.csv`, `scores_per_question.csv`, and `judgments.jsonl` files in this directory to verify the findings. 

*Note: The 41 QA evaluation bank and expert review (both rounds) are available in the `41 QA Evaluation Bank & Expert Review (Round 1 & 2).docx` file located in the repository root.*

---

## Citation
*Placeholder for double-blind review. Citation information will be added in the camera-ready version.*

## License
The dataset and model artifacts are subject to their respective upstream licenses. The Gemma model weights are governed by the [Gemma Terms of Use](https://ai.google.dev/gemma/terms). Derivative models must comply with those terms and must not be used to provide unsupervised medical advice in place of qualified professionals.
