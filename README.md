#### CVDD Text Anomaly Detection — Extended Experiments

This repository contains all scripts for dataloaders, notebooks, logs, files used to reproduce and extend the CVDD (Context Vector Data Description) experiments for text anomaly detection.

Repository Structure
src/datasets/

Contains Python scripts for dataset loading and preprocessing:

wikipedia_topic_mix.py — Builds and preprocesses the Wikipedia Topic Mix dataset for one-class classification, handling scraping, tokenization, and TF-IDF weighting.

bbc2.py — Defines the BBC2 Dataset class to load and preprocess BBC news articles (business, entertainment, politics, sport, tech) with support for spaCy or BERT tokenization and text cleaning.
