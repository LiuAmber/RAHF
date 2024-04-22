# RAHF: Representation Alignment from Human Feedback

## Intruduction

This repo includes a reference implementation of the RAHF for training language models from preference data, as described in the paper[Aligning Large Language Models with Human Preferences through Representation Engineering](https://arxiv.org/abs/2312.15997)

The RAHF pipeline has three stages:

- **Step 0**: Using the [HH-RLHF](https://huggingface.co/datasets/Dahoas/full-hh-rlhf) dataset, perform Supervised Fine-Tuning (SFT) to enable the model with instruction following and conversational abilities.
- **Step 1**: Fine-tune the model using the preference dataset [Ultrafeedback](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned), instructing the model to understand human preferences.
- **Step 2**: Collecting activity patterns and constructing a model with LoRA to fit these patterns.

The files in this repo are:

- `code/step0/data_process.py`：Split the dataset into PPO, RM, and test sets. For reproducibility, we have uploaded the split datasets used in our experiments to the `data` folder.
- `code/step1/DUAL_step1.py`：Perform SFT on the model using the preference dataset. This part corresponds to the code implementation of part **3.1.2** in our paper.
- `code/step1/SCIT_step1.py`：Perform Hindsight on the model using the preference dataset. This part corresponds to the code implementation of part **3.1.1** in our paper.
- `cde/step2/RAHF.py`: This part corresponds to the code implementation of part **3.1.3** in our paper.

## Quickstart

### Install

```bash
pip install -r requirements.txt
```

### Step0:  Using the [HH-RLHF](https://huggingface.co/datasets/Dahoas/full-hh-rlhf) dataset, perform Supervised Fine-Tuning (SFT) 

```bash
cd code
bash bash/SFT-step0.sh
```

### Step1: Instructing LLMs on Human Preferences 

for **SCIT**

```
cd code
bash bash/SCIT-step1.sh
```

for **DUAL**

```
cd code
bash bash/DUAL-step1.sh
```

### Step2: Constructing Final Models 

for **SCIT**

```
cd code
bash bash/SCIT-step2.sh
```

for **DUAL**

```
cd code
bash bash/DUAL-step2.sh
```

### 

### Evaluation

for open llm Evaluation, we use **[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)**.

for Alpaca-Eval, please see [alpaca_eval](https://github.com/tatsu-lab/alpaca_eval).

for MT-Bench, please see **[FastChat](https://github.com/lm-sys/FastChat)**.



