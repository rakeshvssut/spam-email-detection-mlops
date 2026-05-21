# Data Preparation

## Goal
Clean and standardize data for training and evaluation.

## Implemented
- Auto-detect text and label columns
- Normalize labels (`spam`/`ham` and numeric variants)
- Split train/test with stratification

## Main code
- `data_pipeline.py`

## Why this stage matters
Without consistent preprocessing, model behavior becomes unstable and non-reproducible.

## Inputs
- Raw dataset

## Outputs
- Clean text series (`X`)
- Encoded target series (`y`)
- Train/test split
