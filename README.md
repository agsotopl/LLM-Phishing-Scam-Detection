# LLM Phishing & Scam Detection

Detects phishing emails and SMS scam messages using OpenAI embeddings and a
logistic regression classifier, with Claude generating a plain-language
explanation for each prediction.

## Pipeline

1. `src/data_load.py` — downloads DIFrauD (phishing + sms configs) raw JSONL
2. `src/preprocess.py` — merges domains, fixes text encoding, dedupes
3. `src/embeddings.py` — embeds messages with OpenAI text-embedding-3-small
4. `src/resplit.py` — clusters near-duplicates and reassigns train/test/validation splits to prevent leakage while preserving class balance
5. `src/classifier.py` — trains logistic regression, saves model + metrics
6. `src/leakage_check.py` — verifies no near-duplicate leakage across splits
7. `src/overfit_check.py` — train/test gap, cross-validation, regularization sweep diagnostics
8. `src/explain.py` — generates a Claude explanation for every test prediction

## Results

| | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Overall | 0.973 | 0.954 | 0.964 | 0.996 | 0.994 |
| Email | 0.976 | 0.970 | 0.973 | 0.998 | 0.997 |
| SMS | 0.952 | 0.849 | 0.898 | 0.987 | 0.954 |

SMS recall lags email recall by roughly 12 points — short messages carry
less semantic signal for the embedding model to work with than full email
bodies.

## Data leakage note

An initial evaluation showed 98%+ accuracy, which triggered a leakage
check: 15.3% of test rows had a near-duplicate template in the training
set. Messages were re-clustered by near-duplicate similarity and reassigned
to splits at the cluster level (stratified by class) before final
evaluation, eliminating leakage while keeping class balance consistent
across train/test/validation.

## Setup

Create and activate a virtual environment, then install dependencies:

    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

Copy the environment template and fill in your API keys:

    cp .env.example .env

Run the pipeline in order:

    python src/data_load.py
    python src/preprocess.py
    python src/embeddings.py
    python src/resplit.py
    python src/classifier.py
    python src/leakage_check.py
    python src/explain.py

## Dataset

[DIFrauD](https://huggingface.co/datasets/redasers/difraud) (University of
Houston), `phishing` and `sms` configs.