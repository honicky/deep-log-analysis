---
language: en
tags:
- log-analysis
- pythia
- hdfs
license: mit
datasets:
- honicky/log-analysis-hdfs-preprocessed
metrics:
- cross-entropy
- perplexity
base_model: EleutherAI/pythia-14m
---

# pythia-14m-hdfs-logs

Fine-tuned Pythia-14m model for HDFS log analysis, specifically for anomaly detection.

## Model Description

This model is fine-tuned from `EleutherAI/pythia-14m` for analyzing HDFS log sequences. It's designed to understand and predict patterns in
HDFS log data so that we can detect anomalies using the perplexity of the log sequence. THhe HDFS sequence is handy because it has labels
so we can use it to validate that the model can predict anomalies. 

We will use this model to understand the ability of a small model to predict anomalies in a specific dataset.  We will study model scale
and experiment with tokenization, intialization, data set size, etc. to find a configuration that is minimal in size and fast, but can
effectively predict anomalies.  We will then attempt build a model that is more robust to different log formats.

## Training Details
- Base model: EleutherAI/pythia-14m
- Dataset: https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1 + preprocessed data at honicky/log-analysis-hdfs-preprocessed
- Batch size: 4
- Max sequence length: 343
- Learning rate: 0.0001
- Training steps: 2110

## Special Tokens
- Added `<|sep|>` token for event ID separation

## Intended Use
This model is intended for:
- Analyzing HDFS log sequences
- Detecting anomalies in log patterns
- Understanding system behavior through log analysis

## Limitations
- Model is specifically trained on HDFS logs and may not generalize to other log formats
- Limited to the context window size of 343 tokens


