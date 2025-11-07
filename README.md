## CVDD Text Anomaly Detection — Extended Experiments

This repository contains all scripts for dataloaders, notebooks, logs, files used to reproduce and extend the CVDD (Context Vector Data Description) experiments for text anomaly detection.

## Repository Structure src/datasets/

Contains Python scripts for dataset loading and processing:

- wikipedia_topic_mix.py — Builds and preprocesses the Wikipedia Topic Mix dataset for one-class classification, handling scraping, tokenization.
- bbc2.py — Defines the BBC2 Dataset class to load and preprocess BBC news articles (business, entertainment, politics, sport, tech) with support for spaCy or BERT tokenization and text cleaning.

## New Data Construction/

Contains Jupyter notebooks (new_data.ipynb) used to create new datasets from external sources.
This notebook perform article retrieval, text cleaning, category filtering, and export of class-wise text files later used by the dataset loaders in src/datasets/.

## log/

Stores all experiment logs. Each subfolder corresponds to a specific run (dataset, embedding, and random seed).

Example: log/20251007-220632_newsgroups20_GloVe_6B_r3_c4_seed1

### Files:
log.txt → Detailed training and evaluation metrics, including AUC scores.

top_words_test.txt → Records attention and context vectors, showing which words received the highest attention weights.

## Code Integration Note:

In main.py, the new dataset names 'BBC2' and 'wikipedia_topic_mix' were added to the dataset selection options.

Without this modification, the training script would raise a “dataset not found” error when trying to run these datasets.

## Usage Instructions

Run the notebooks in New Data Construction/ to generate or refresh datasets.

Use the dataset loaders in src/datasets/ (wikipedia_topic_mix.py or bbc2.py) to prepare data for training.

Confirm that main.py includes 'BBC2' and 'wikipedia_topic_mix' in the dataset list.

### Verify results:

Open log/<timestamped_folder>/log.txt for AUC values.

Open top_words_test.txt for attention and context vector outputs.
Example Log Path
log/20251007-220632_newsgroups20_GloVe_6B_r3_c4_seed1/
├── log.txt              # AUC metrics and training stats
├── top_words_test.txt   # Attention + context vectors
