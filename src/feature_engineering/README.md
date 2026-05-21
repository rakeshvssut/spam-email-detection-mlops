# Feature Engineering

## Goal
Transform text into numerical features for machine learning.

## Current approach
- TF-IDF vectorization in training pipeline

## Why this stage matters
Text models cannot consume raw strings directly; feature extraction turns language patterns into learnable signals.

## Future improvements
- Stopword tuning
- N-gram exploration
- Character-level features
- Domain-specific token filters

## Inputs
- Prepared text

## Outputs
- Sparse numerical feature vectors
